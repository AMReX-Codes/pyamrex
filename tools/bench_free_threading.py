#!/usr/bin/env python3
"""Free-threading scaling benchmark for pyAMReX.

Measures how a per-box Python compute kernel scales with the number of Python
threads. On a free-threaded interpreter the same binary can be run with the GIL
off and on, which makes the comparison exact -- same build, same machine, one
environment variable apart::

    PYTHON_GIL=0 python tools/bench_free_threading.py --json off.json
    PYTHON_GIL=1 python tools/bench_free_threading.py --json on.json
    python tools/bench_free_threading.py --compare off.json on.json

Two kernels are measured because they answer different questions:

``numpy``
    A vectorised expression. NumPy releases the GIL inside the ufunc, so even a
    GIL build scales somewhat; this is the pattern most pyAMReX code already
    uses.
``python``
    A Python-level loop over the same data. This holds the GIL for its whole
    duration, so it cannot scale at all on a GIL build. This is where
    free-threading actually changes what is possible.

Two iteration modes:

``snapshot`` (default)
    One serial MFIter pass collects the per-box Array4 views, then the thread
    pool runs the kernels. Works on any build.
``mfiter``
    Each thread drives its own MFIter and picks its boxes by index. Requires
    the free-threading fixes to AMReX (``MFIter::depth`` and the FabArrayBase
    caches); on an unfixed build this aborts on
    "Nested or multiple active MFIters is not supported by default".
"""

import argparse
import json
import os
import statistics
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import amrex.space3d as amr

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def kernel_numpy(view, work):
    """Vectorised update; NumPy releases the GIL for the ufuncs."""
    for _ in range(work):
        view *= 1.0000001
        view += 1.0


def kernel_python(view, work):
    """Python-level loop over the same data; holds the GIL throughout."""
    flat = view.reshape(-1)
    n = flat.size
    stride = max(1, n // 4096)
    for _ in range(work):
        acc = 0.0
        for i in range(0, n, stride):
            acc += float(flat[i]) * 1.5
        flat[0] = acc


KERNELS = {"numpy": kernel_numpy, "python": kernel_python}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def make_multifab(ncell, max_grid_size, ncomp):
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(*([ncell - 1] * 3)))
    ba = amr.BoxArray(domain)
    ba.max_size(max_grid_size)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, ncomp, 0)
    mf.set_val(1.0)
    return mf


def run_snapshot(mf, kernel, work, nthreads):
    """Serial MFIter pass to collect views, then run the kernels in a pool."""
    views = [mf.array(mfi).to_xp(copy=False, order="F") for mfi in mf]
    barrier = threading.Barrier(nthreads)

    def worker(t):
        barrier.wait(timeout=300)
        for i in range(t, len(views), nthreads):
            kernel(views[i], work)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        list(pool.map(worker, range(nthreads)))
    return time.perf_counter() - t0


def run_mfiter(mf, kernel, work, nthreads):
    """Each thread drives its own MFIter and claims boxes round-robin."""
    barrier = threading.Barrier(nthreads)

    def worker(t):
        barrier.wait(timeout=300)
        for mfi in mf:
            if mfi.index % nthreads != t:
                continue
            kernel(mf.array(mfi).to_xp(copy=False, order="F"), work)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        list(pool.map(worker, range(nthreads)))
    return time.perf_counter() - t0


RUNNERS = {"snapshot": run_snapshot, "mfiter": run_mfiter}


def bench(args):
    runner = RUNNERS[args.mode]
    results = {}
    for kernel_name in args.kernels:
        kernel = KERNELS[kernel_name]
        per_threads = {}
        for nthreads in args.threads:
            mf = make_multifab(args.ncell, args.max_grid_size, args.ncomp)
            runner(mf, kernel, args.work, nthreads)  # warm-up
            samples = [
                runner(mf, kernel, args.work, nthreads) for _ in range(args.repeats)
            ]
            per_threads[nthreads] = min(samples)
            print(
                f"  {kernel_name:>6} {nthreads:>3} threads: "
                f"{min(samples):8.4f} s  (median {statistics.median(samples):.4f} s)",
                flush=True,
            )
            mf.clear()
        results[kernel_name] = per_threads
    return results


def overhead(args):
    """Single-threaded cost of an MFIter + fill_boundary loop.

    This is the A/B for the mutexes added to AMReX: run it on an unmodified
    checkout, then again on the patched one.
    """
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(*([args.ncell - 1] * 3)))
    ba = amr.BoxArray(domain)
    ba.max_size(args.max_grid_size)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 1)
    mf.set_val(1.0)

    def loop(n):
        t0 = time.perf_counter()
        for _ in range(n):
            for mfi in mf:
                mf.array(mfi)
            mf.fill_boundary()
        return time.perf_counter() - t0

    loop(2)  # warm-up
    samples = [loop(args.repeats) for _ in range(5)]
    per_iter = min(samples) / args.repeats
    print(f"  MFIter + fill_boundary: {per_iter * 1e3:.4f} ms / iteration")
    return {"mfiter_fill_boundary_s": per_iter}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def describe_runtime():
    return {
        "python": sys.version.split()[0],
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "cpu_count": os.cpu_count(),
        "amrex_version": amr.__version__,
        "gpu_backend": amr.Config.gpu_backend,
        "precision": amr.Config.precision,
    }


def print_table(results):
    for kernel_name, per_threads in results.items():
        base = per_threads[min(per_threads)]
        print(f"\n{kernel_name} kernel")
        print(f"  {'threads':>7}  {'time [s]':>9}  {'speedup':>8}  {'efficiency':>10}")
        for nthreads in sorted(per_threads):
            t = per_threads[nthreads]
            print(
                f"  {nthreads:>7}  {t:>9.4f}  {base / t:>8.2f}x  "
                f"{100 * base / t / nthreads:>9.0f}%"
            )


def compare(paths):
    """Print a side-by-side table from result files written by ``--json``."""
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    labels = ["GIL on" if run["runtime"]["gil_enabled"] else "GIL off" for run in runs]

    for kernel_name in sorted(runs[0]["results"]):
        print(f"\n{kernel_name} kernel -- time [s] and speedup vs. 1 thread")
        header = "  threads"
        for label in labels:
            header += f"  {label:>16}"
        print(header)
        threads = sorted(int(k) for k in runs[0]["results"][kernel_name])
        for nthreads in threads:
            row = f"  {nthreads:>7}"
            for run in runs:
                per = run["results"][kernel_name]
                t = per[str(nthreads)]
                base = per[str(threads[0])]
                row += f"  {t:>9.4f} {base / t:>5.2f}x"
            print(row)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ncell", type=int, default=128, help="cells per dimension")
    p.add_argument("--max-grid-size", type=int, default=32, help="box size")
    p.add_argument("--ncomp", type=int, default=1, help="components per box")
    p.add_argument("--work", type=int, default=1, help="kernel repetitions per box")
    p.add_argument("--repeats", type=int, default=5, help="timed samples")
    p.add_argument("--threads", type=int, nargs="+", default=None)
    p.add_argument("--kernels", nargs="+", default=list(KERNELS), choices=list(KERNELS))
    p.add_argument("--mode", default="snapshot", choices=list(RUNNERS))
    p.add_argument("--json", help="write results here")
    p.add_argument("--overhead", action="store_true", help="single-thread locking A/B")
    p.add_argument("--compare", nargs="+", help="print a table from result files")
    args = p.parse_args(argv)

    if args.compare:
        compare(args.compare)
        return 0

    if args.threads is None:
        ncpu = os.cpu_count() or 1
        args.threads = [n for n in (1, 2, 4, 8, 14, 20) if n <= ncpu] or [1]

    amr.initialize(
        [
            "amrex.verbose=0",
            "tiny_profiler.enabled=0",
            "amrex.throw_exception=1",
            "amrex.signal_handling=0",
        ]
    )
    try:
        runtime = describe_runtime()
        print("runtime:", json.dumps(runtime))
        if args.overhead:
            payload = {"runtime": runtime, "overhead": overhead(args)}
        else:
            results = bench(args)
            print_table(results)
            payload = {"runtime": runtime, "args": vars(args), "results": results}
        if args.json:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nwrote {args.json}")
    finally:
        amr.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
