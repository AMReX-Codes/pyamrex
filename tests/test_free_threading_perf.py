# -*- coding: utf-8 -*-
"""Scaling regression tests for free-threaded CPython (PEP 703).

These are timing-based and therefore off by default; a loaded machine would
make them flap. Enable them explicitly::

    PYAMREX_BENCH=1 pytest tests/test_free_threading_perf.py -s

For the full sweep and the GIL-on/GIL-off comparison, use
``tools/bench_free_threading.py`` instead -- this file only asserts the floor
that a regression would break through.
"""

import os
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import amrex.space3d as amr


def _amrex_is_free_threading_safe():
    """Same AMReX capability probe the functional suite uses."""
    from test_free_threading import AMREX_IS_FREE_THREADING_SAFE

    return AMREX_IS_FREE_THREADING_SAFE


FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
GIL_ENABLED = getattr(sys, "_is_gil_enabled", lambda: True)()
BENCH_ENABLED = os.environ.get("PYAMREX_BENCH") == "1"

#: Threads to scale to. Kept modest so the floor holds on a busy CI runner.
NTHREADS = 4

#: Minimum speedup at NTHREADS for a GIL-bound kernel with the GIL disabled.
#: Linear would be 4.0; 2.0 leaves generous headroom for noise and for the
#: interpreter's own free-threading overhead.
MIN_SPEEDUP = 2.0

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not BENCH_ENABLED, reason="set PYAMREX_BENCH=1 to run"),
]


def python_kernel(view, work=2):
    """Python-level loop over the data: holds the GIL for its whole duration."""
    flat = view.reshape(-1)
    n = flat.size
    stride = max(1, n // 4096)
    for _ in range(work):
        acc = 0.0
        for i in range(0, n, stride):
            acc += float(flat[i]) * 1.5
        flat[0] = acc


def time_threaded(views, nthreads):
    """Best-of-3 wall time for running python_kernel over ``views``."""
    barrier = threading.Barrier(nthreads)

    def worker(t):
        barrier.wait(timeout=300)
        for i in range(t, len(views), nthreads):
            python_kernel(views[i])

    def once():
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            list(pool.map(worker, range(nthreads)))
        return time.perf_counter() - t0

    once()  # warm-up
    return min(once() for _ in range(3))


@pytest.fixture(scope="function")
def bench_views():
    """Per-box zero-copy views of an over-decomposed MultiFab."""
    ncell, max_grid_size = 128, 32
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(*([ncell - 1] * 3)))
    ba = amr.BoxArray(domain)
    ba.max_size(max_grid_size)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(1.0)
    views = [mf.array(mfi).to_xp(copy=False, order="F") for mfi in mf]
    assert len(views) >= NTHREADS
    yield views
    del views
    mf.clear()


@pytest.mark.skipif(
    not FREE_THREADED or GIL_ENABLED,
    reason="needs a free-threaded interpreter with the GIL actually disabled",
)
@pytest.mark.skipif(
    (os.cpu_count() or 1) < NTHREADS, reason=f"needs >= {NTHREADS} cores"
)
def test_python_kernel_scales_without_the_gil(bench_views):
    """A GIL-bound per-box kernel must actually get faster with more threads.

    On a GIL build this speedup is ~1.0 by construction, which is the whole
    point of pyamrex#612.
    """
    t1 = time_threaded(bench_views, 1)
    tn = time_threaded(bench_views, NTHREADS)
    speedup = t1 / tn
    print(
        f"\npython kernel: 1 thread {t1:.4f} s, {NTHREADS} threads {tn:.4f} s "
        f"-> {speedup:.2f}x ({100 * speedup / NTHREADS:.0f}% efficiency)"
    )
    assert speedup >= MIN_SPEEDUP, (
        f"expected >= {MIN_SPEEDUP}x with the GIL disabled, got {speedup:.2f}x"
    )


@pytest.mark.skipif(
    (os.cpu_count() or 1) < NTHREADS, reason=f"needs >= {NTHREADS} cores"
)
@pytest.mark.skipif(
    not _amrex_is_free_threading_safe(),
    reason="AMReX predates the host-thread-safety work (AMReX-Codes/amrex#5615)",
)
def test_per_thread_mfiter_scales(boxarr, distmap):
    """The same, but with each thread driving its own MFIter.

    This is the pattern the AMReX free-threading fixes unlock; before them the
    second concurrent MFIter aborts the process.
    """
    mf = amr.MultiFab(boxarr, distmap, 1, 0)
    mf.set_val(1.0)

    def run(nthreads):
        barrier = threading.Barrier(nthreads)

        def worker(t):
            barrier.wait(timeout=300)
            for mfi in mf:
                if mfi.index % nthreads == t:
                    python_kernel(mf.array(mfi).to_xp(copy=False, order="F"))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            list(pool.map(worker, range(nthreads)))
        return time.perf_counter() - t0

    run(1)  # warm-up
    t1 = min(run(1) for _ in range(3))
    tn = min(run(NTHREADS) for _ in range(3))
    print(f"\nper-thread MFIter: {t1:.4f} s -> {tn:.4f} s ({t1 / tn:.2f}x)")

    if FREE_THREADED and not GIL_ENABLED:
        assert t1 / tn >= MIN_SPEEDUP
    mf.clear()
