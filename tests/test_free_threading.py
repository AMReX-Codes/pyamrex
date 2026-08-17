# -*- coding: utf-8 -*-
"""Concurrency tests for free-threaded CPython (PEP 703).

Threads are spawned *inside* each test, never across tests: the autouse
``amrex_init`` fixture in ``conftest.py`` wraps every test in
``amr.initialize()`` / ``amr.finalize()``, which is process-global state and
stays on one thread.

Every test is written so that a data race shows up as a *wrong value*, not just
as a crash. Run them repeatedly to shake out timing-dependent races::

    pytest tests/test_free_threading.py --count=50

Most of these drive one ``MFIter`` per thread, which needs an AMReX whose
``MFIter::depth`` is per-thread (AMReX-Codes/amrex#5615). Older AMReX aborts
the second concurrent iterator with "Nested or multiple active MFIters is not
supported by default", so those tests are skipped there -- see
``AMREX_IS_FREE_THREADING_SAFE``. This is about the AMReX underneath, not about whether
the GIL is enabled: they are equally valid, if serialized, on a GIL build.
"""

import os
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import amrex.space3d as amr

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

#: Enough threads to expose races without oversubscribing a CI runner.
NTHREADS = max(2, min(8, os.cpu_count() or 2))

#: Seconds a worker may wait for its peers at the start barrier.
BARRIER_TIMEOUT = 120.0


def _probe_concurrent_mfiter():
    """Does the AMReX we are linked against allow one MFIter per thread?

    Two iterators on *different* threads, both alive at once -- nesting them on
    one thread is a different thing and is still rejected. Probed in a
    subprocess because on an AMReX without AMReX-Codes/amrex#5615 the failure
    leaves the global depth counter non-zero, which would poison every later
    test in this process.
    """
    probe = """
import threading
import amrex.space3d as amr

amr.initialize(['amrex.verbose=0', 'amrex.throw_exception=1',
                'amrex.signal_handling=0', 'tiny_profiler.enabled=0'])
bx = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(7, 7, 7))
ba = amr.BoxArray(bx)
mf = amr.MultiFab(ba, amr.DistributionMapping(ba), 1, 0)

errors = []
barrier = threading.Barrier(2)


def work(i):
    # Ordered, not simultaneous: thread 0's iterator is provably alive before
    # thread 1 builds its own. Racing both constructions would let two
    # non-atomic ++depth from 0 lose an update on an old AMReX, so both would
    # read depth==1 and the probe would wrongly report success.
    try:
        if i == 0:
            it = amr.MFIter(mf)  # noqa: F841  -- held alive across the barriers
            barrier.wait(timeout=60)
            barrier.wait(timeout=60)
        else:
            barrier.wait(timeout=60)
            it = amr.MFIter(mf)  # noqa: F841  -- must succeed alongside 0's
            barrier.wait(timeout=60)
    except Exception as e:       # noqa: BLE001
        errors.append(e)


threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
if not errors:
    print('yes')
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0 and proc.stdout.strip().endswith("yes")


#: Whether AMReX has the host-thread-safety work (AMReX-Codes/amrex#5615).
#: Probed via one-MFIter-per-thread, which is the cheapest observable symptom;
#: the tests below depend on the whole of it, not only on that piece.
AMREX_IS_FREE_THREADING_SAFE = _probe_concurrent_mfiter()

needs_amrex_free_threading = pytest.mark.skipif(
    not AMREX_IS_FREE_THREADING_SAFE,
    reason="AMReX predates the host-thread-safety work (AMReX-Codes/amrex#5615)",
)


def run_concurrently(fn, nthreads=NTHREADS):
    """Run ``fn(i)`` on ``nthreads`` threads, started as close to simultaneously
    as a barrier allows.

    Returns the per-thread results in thread-index order. The first exception
    raised in any worker is re-raised here, so a plain ``assert`` inside ``fn``
    works as usual.
    """
    barrier = threading.Barrier(nthreads)

    def worker(i):
        barrier.wait(timeout=BARRIER_TIMEOUT)
        return fn(i)

    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        futures = [pool.submit(worker, i) for i in range(nthreads)]
        return [f.result() for f in futures]


def valid_slices(arr_shape, n_grow_vect):
    """Slices into the valid (non-ghost) region of a 4-axis (x,y,z,n) Array4
    view, for any AMREX_SPACEDIM. Same helper as in test_fill_domain_boundary.py.
    """
    sd = amr.Config.spacedim
    ng = [n_grow_vect[d] if d < sd else 0 for d in range(3)] + [0]
    return tuple(slice(g, s - g) for g, s in zip(ng, arr_shape))


def count_equal(arr, value):
    """Number of entries equal to ``value``, as a host int (CPU and GPU alike)."""
    return int((arr == value).sum())


def make_particle_container(std_geometry, distmap, boxarr):
    """Legacy AoS+SoA container, managed memory on GPU (as in test_particleContainer.py)."""
    if amr.Config.have_gpu:
        return amr.ParticleContainer_2_1_3_1_managed(std_geometry, distmap, boxarr)
    return amr.ParticleContainer_2_1_3_1_default(std_geometry, distmap, boxarr)


def make_particle_init():
    myt = amr.ParticleInitType_2_1_3_1()
    myt.real_struct_data = [0.5, 0.6]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2, 0.3]
    myt.int_array_data = [1]
    return myt


# ---------------------------------------------------------------------------
# the free-threading opt-in itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded CPython build")
def test_module_does_not_reenable_the_gil():
    """Importing pyAMReX must not switch the GIL back on.

    Without ``py::mod_gil_not_used()`` on the ``PYBIND11_MODULE``, CPython
    re-enables the GIL process-wide at import time and the free-threaded
    interpreter gives no benefit at all (pyamrex#612).

    Checked in a subprocess with a clean environment: ``PYTHON_GIL`` and
    ``-X gil`` force the outcome either way, so testing in-process would only
    report how *this* pytest run happened to be launched.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import amrex.space3d; print(sys._is_gil_enabled())",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        "importing pyAMReX re-enabled the GIL process-wide; the module is "
        f"missing py::mod_gil_not_used().\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------


@needs_amrex_free_threading
def test_concurrent_multifab_lifetime(boxarr, distmap):
    """Concurrent MultiFab construction and destruction.

    Every FabArray ctor/dtor runs ``FabArrayBase::addThisBD``/``clearThisBD``,
    which mutate the process-global ``m_BD_count`` map and, on the last
    reference, cascade into all the cache flushers.
    """

    def work(t):
        value = float(t) + 1.0
        for _ in range(20):
            mf = amr.MultiFab(boxarr, distmap, 1, 0)
            mf.set_val(value)
            for mfi in mf:
                arr = mf.array(mfi).to_xp(copy=False, order="F")
                assert float(arr.min()) == value
                assert float(arr.max()) == value
            mf.clear()
        return t

    # the real checks are the asserts inside work(); this only pins down that
    # every thread ran and results came back in thread order
    assert run_concurrently(work) == list(range(NTHREADS))


@needs_amrex_free_threading
def test_concurrent_mfiter_distinct_multifabs(boxarr, distmap):
    """Each thread iterates its own MultiFab -- the plainest data-parallel case."""

    def work(t):
        value = float(t) + 1.0
        mf = amr.MultiFab(boxarr, distmap, 1, 0)
        mf.set_val(0.0)
        ncells = 0
        for mfi in mf:
            arr = mf.array(mfi).to_xp(copy=False, order="F")
            arr[...] = value
            ncells += arr.size
        assert sum(
            count_equal(mf.array(mfi).to_xp(copy=False), value) for mfi in mf
        ) == (ncells)
        return ncells

    counts = run_concurrently(work)
    assert len(set(counts)) == 1, counts


@needs_amrex_free_threading
def test_concurrent_mfiter_same_multifab(mfab):
    """Each thread drives its *own* MFIter over one shared MultiFab.

    AMReX asserts only one live MFIter per process (``MFIter::depth``, a plain
    global ``int``), and the tile-array cache lookup behind every MFIter is
    guarded only by ``#pragma omp critical`` -- which compiles away in a
    non-OpenMP build. Threads here partition the boxes round-robin by index, so
    no two write to the same data.
    """
    mfab.set_val(-1.0)
    nboxes = sum(1 for _ in mfab)

    def work(t):
        touched = 0
        for mfi in mfab:
            if mfi.index % NTHREADS != t:
                continue
            arr = mfab.array(mfi).to_xp(copy=False, order="F")
            arr[...] = float(t) + 1.0
            touched += 1
        return touched

    assert sum(run_concurrently(work)) == nboxes

    for mfi in mfab:
        arr = mfab.array(mfi).to_xp(copy=False, order="F")
        expected = float(mfi.index % NTHREADS) + 1.0
        assert float(arr.min()) == expected, f"box {mfi.index}"
        assert float(arr.max()) == expected, f"box {mfi.index}"


@needs_amrex_free_threading
def test_concurrent_fill_boundary(boxarr, distmap):
    """Concurrent ``fill_boundary`` on distinct MultiFabs with identical shape.

    ``FabArrayBase::getFB`` looks up and inserts into a process-global multimap
    with no lock at all. Identical shapes mean identical cache keys, i.e. the
    worst case for that cache.
    """
    sentinel = -42.0

    def build_and_fill(value):
        mf = amr.MultiFab(boxarr, distmap, 1, 1)
        mf.set_val(sentinel)
        for mfi in mf:
            arr = mf.array(mfi).to_xp(copy=False, order="F")
            arr[valid_slices(arr.shape, mf.n_grow_vect)] = value
        mf.fill_boundary()
        return mf

    def filled_cells(mf, value):
        total = 0
        for mfi in mf:
            arr = mf.array(mfi).to_xp(copy=False, order="F")
            n_value = count_equal(arr, value)
            # a cross-thread leak would show up as a third value
            assert n_value + count_equal(arr, sentinel) == arr.size
            total += n_value
        return total

    reference = filled_cells(build_and_fill(1.0), 1.0)
    assert reference > 0

    def work(t):
        value = float(t) + 1.0
        return filled_cells(build_and_fill(value), value)

    assert run_concurrently(work) == [reference] * NTHREADS


@needs_amrex_free_threading
def test_concurrent_sum_boundary(boxarr, distmap):
    """Concurrent ``sum_boundary`` -- exercises the global copy-plan cache
    (``FabArrayBase::getCPC``), which is likewise unlocked."""

    def build_and_sum(value):
        mf = amr.MultiFab(boxarr, distmap, 1, 1)
        mf.set_val(value)
        mf.sum_boundary(amr.Periodicity())
        return mf

    def total(mf):
        return sum(
            float(mf.array(mfi).to_xp(copy=False, order="F").sum()) for mfi in mf
        )

    reference = total(build_and_sum(1.0))
    assert reference > 0

    def work(t):
        # a per-thread value, so a thread reading another's buffer is visible
        value = float(t) + 1.0
        return total(build_and_sum(value)) / value

    for got in run_concurrently(work):
        assert got == pytest.approx(reference)


# ---------------------------------------------------------------------------
# runtime parameters
# ---------------------------------------------------------------------------


@needs_amrex_free_threading
def test_concurrent_parmparse_query():
    """Concurrent queries against the process-global ParmParse table.

    A ParmParse *query* is not read-only: it bumps the entry's use count and
    writes its type hint and last value, all through ``mutable`` members of a
    table entry shared by every ParmParse object in the process. Recording the
    last value *grows a vector* the first time an entry is queried, so the keys
    below are deliberately queried for the first time from several threads at
    once -- a warm-up query on the main thread would pre-size that vector and
    hide the race.
    """
    nkeys = 32
    expected = {f"key{i:02d}": 100 + i for i in range(nkeys)}

    pp_setup = amr.ParmParse("ft")
    for name, value in expected.items():
        pp_setup.add(name, value)

    def work(t):
        pp = amr.ParmParse("ft")
        seen = {}
        for _ in range(10):
            for name, value in expected.items():
                found, got = pp.query_int(name)
                assert found, name
                assert got == value, (name, got, value)
                seen[name] = got
        return seen

    assert run_concurrently(work) == [expected] * NTHREADS


# ---------------------------------------------------------------------------
# particles
# ---------------------------------------------------------------------------


@needs_amrex_free_threading
def test_concurrent_particle_containers(std_geometry, distmap, boxarr):
    """Each thread builds, fills and redistributes its own ParticleContainer.

    Touches the global particle-id counter (a bare ``++`` without OpenMP), the
    particle-tile runtime pointer caches, and the lazily initialised statics in
    ``ParticleContainerBase``.

    Note that the particles are placed deterministically rather than with
    ``init_random``: that call *reseeds the process-global RNG* (it forwards to
    ``amrex::InitRandom``), so it belongs on one thread before the workers
    start -- see :ref:`usage-threading`.
    """
    npart = int(std_geometry.domain.num_pts)

    def work(t):
        pc = make_particle_container(std_geometry, distmap, boxarr)
        pc.init_one_per_cell(0.5, 0.5, 0.5, make_particle_init())
        pc.redistribute()
        assert pc.OK()
        return pc.total_number_of_particles()

    assert run_concurrently(work) == [npart] * NTHREADS


@needs_amrex_free_threading
def test_concurrent_soa_views(std_geometry, distmap, boxarr):
    """Concurrent zero-copy views onto particle SoA data."""

    def work(t):
        pc = make_particle_container(std_geometry, distmap, boxarr)
        pc.init_one_per_cell(0.5, 0.5, 0.5, make_particle_init())
        seen = 0
        for lvl in range(pc.finest_level + 1):
            for tile in pc.get_particles(lvl).values():
                real_arrays = tile.get_struct_of_arrays().get_real_data()
                for pod in real_arrays:
                    if pod.size() == 0:
                        continue  # to_xp() rejects empty vectors
                    seen += int(pod.to_xp(copy=False).size)
        return seen

    counts = run_concurrently(work)
    assert len(set(counts)) == 1, counts
    assert counts[0] > 0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="requires AMREX_SPACEDIM = 3")
@needs_amrex_free_threading
def test_concurrent_plotfile_read(tmp_path):
    """Several threads read the same plotfile at once.

    AMReX caches an open ``ifstream`` per file name. That cache used to be
    process-global, so two threads reading the same file shared one stream --
    and therefore one read position.
    """
    filename = str(tmp_path / "test_plt_free_threading")
    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])
    ba = amr.BoxArray(domain_box)
    ba.max_size(16)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(3.5)
    amr.write_single_level_plotfile(
        filename, mf, amr.Vector_string(["density"]), geom, 1.0, 200
    )

    def work(t):
        plt = amr.PlotFileData(filename)
        field = plt.get(0, "density")
        total = 0.0
        for mfi in field:
            arr = field.array(mfi).to_xp(copy=False, order="F")
            assert count_equal(arr, 3.5) == arr.size
            total += float(arr.sum())
        return total

    totals = run_concurrently(work)
    assert totals == pytest.approx([3.5 * 32**3] * NTHREADS)


# ---------------------------------------------------------------------------
# binding layer
# ---------------------------------------------------------------------------


def test_concurrent_pyobject_churn():
    """Heavy creation and destruction of small bound objects from many threads.

    This is a stress test of pybind11's process-global instance registry rather
    than of AMReX; it is here so a pybind11 regression is caught by pyAMReX's
    own suite.
    """
    nreps = 2000

    def work(t):
        acc = 0
        for k in range(nreps):
            lo = amr.IntVect(0, 0, 0)
            hi = amr.IntVect(t + 1, k % 7 + 1, 3)
            box = amr.Box(lo, hi)
            acc += box.num_pts
        return acc

    def reference(t):
        return sum(
            amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(t + 1, k % 7 + 1, 3)).num_pts
            for k in range(nreps)
        )

    assert run_concurrently(work) == [reference(t) for t in range(NTHREADS)]


@needs_amrex_free_threading
def test_concurrent_array_views(boxarr, distmap):
    """Concurrent creation and release of zero-copy array views.

    pyAMReX keeps a process-global count of outstanding DLPack exports, and each
    view holds a reference back to its Python owner.
    """

    def work(t):
        value = float(t) + 1.0
        mf = amr.MultiFab(boxarr, distmap, 1, 0)
        mf.set_val(value)
        for _ in range(50):
            views = [mf.array(mfi).to_xp(copy=False) for mfi in mf]
            assert all(float(v.min()) == value for v in views)
            del views
        return t

    # the real checks are the asserts inside work(); this only pins down that
    # every thread ran and results came back in thread order
    assert run_concurrently(work) == list(range(NTHREADS))
