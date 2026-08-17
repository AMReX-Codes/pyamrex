# -*- coding: utf-8 -*-

import itertools
import os
import platform
import subprocess
import sys

import numpy as np
import pytest

try:
    import amrex.space3d as amr
except ImportError:
    try:
        import amrex.space2d as amr
    except ImportError:
        try:
            import amrex.space1d as amr
        except ImportError:
            raise ImportError("AMReX: No 1D, 2D or 3D module found!")

# Import calls MPI_Initialize, if not called already
if amr.Config.have_mpi:
    from mpi4py import MPI  # noqa

# base path for input files
basepath = os.getcwd()


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
        # Break the barrier so the other thread stops waiting immediately. On
        # an AMReX without the fix this is the expected path, and without the
        # abort the surviving thread sits out the full timeout -- 60 s added to
        # every CI run, since dependencies.json still pins a pre-#5615 AMReX.
        barrier.abort()


threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
del mf, ba
amr.finalize()
if not errors:
    print('yes')
"""
    # Drop the MPI launcher's environment. The suite itself is normally run
    # under `mpiexec -n 1`, and a child that inherits those variables tries to
    # join the parent's job instead of starting as a singleton -- which either
    # hangs or takes the whole job down with it. Without them, the probe is an
    # ordinary one-rank process.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_", "MPIR_CVAR_"))
        and k not in ("MPI_LOCALRANKID", "MPI_LOCALNRANKS")
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().endswith("yes")


@pytest.fixture(scope="session")
def needs_amrex_free_threading():
    """Skip unless AMReX has the host-thread-safety work.

    Probed via one-MFIter-per-thread, the cheapest observable symptom of
    AMReX-Codes/amrex#5615; the tests that ask for this depend on the whole of
    that work, not only on that piece.

    A session fixture rather than a module-level constant on two counts. The
    probe costs a subprocess ``amrex.initialize``, and running it at conftest
    import charged that to every pytest session, including ones with no
    concurrency test in them. And a fixture means the test modules need not
    import conftest, which only resolves under pytest's default ``prepend``
    import mode.
    """
    if not _probe_concurrent_mfiter():
        pytest.skip(
            "AMReX predates the host-thread-safety work (AMReX-Codes/amrex#5615)"
        )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "perf: timing-based scaling benchmark; set PYAMREX_BENCH=1 to run",
    )


@pytest.fixture(scope="function")
def make_real_array4():
    """Create an Array4 of ones matching the compiled amrex::Real precision."""

    def make(shape):
        if amr.Config.precision == "SINGLE":
            return amr.Array4_float(np.ones(shape, dtype=np.float32))
        else:
            return amr.Array4_double(np.ones(shape, dtype=np.float64))

    return make


@pytest.fixture(scope="function")
def assert_keeps_python_alive():
    """Assert that a pybind11 object keeps its Python owner alive."""
    if platform.python_implementation() != "CPython":
        pytest.skip("sys.getrefcount-based lifetime checks are CPython-specific")

    def check(owner, make_view):
        before = sys.getrefcount(owner)
        view = make_view()
        assert sys.getrefcount(owner) > before
        return view

    return check


@pytest.fixture(autouse=True, scope="function")
def amrex_init(tmpdir):
    with tmpdir.as_cwd():
        amr.initialize(
            [
                # print AMReX status messages
                # consider also 0 (silent) and 2 (FabArray and TileArray/FB/Copy/FillPatch/CrsFineCache usage)
                "amrex.verbose=1",
                # disable verbose profiler plots at the end of each test
                "tiny_profiler.enabled=0",
                # throw exceptions and create core dumps instead of
                # AMReX backtrace files: allows to attach to
                # debuggers
                "amrex.throw_exception=1",
                "amrex.signal_handling=0",
                # abort GPU runs if out-of-memory instead of swapping to host RAM
                # "abort_on_out_of_gpu_memory=1",
                # avoid managed memory unless explicitly used
                "amrex.the_arena_is_managed=0",
                # allocate GPU memory on-demand instead of pre-allocating 3/4th
                # to enable parallel test runs on the same GPU
                # https://amrex-codes.github.io/amrex/docs_html/RuntimeParameters.html?highlight=arena#memory
                "amrex.the_arena_init_size=0",
            ]
        )
        yield
        amr.finalize()


@pytest.fixture(scope="function")
def std_real_box():
    """Standard RealBox for common problem domains"""
    rb = amr.RealBox(0, 0, 0, 1.0, 1.0, 1.0)
    return rb


@pytest.fixture(scope="function")
def std_box():
    """Standard Box for tests"""
    bx = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    return bx


@pytest.fixture(scope="function")
def std_geometry(std_box, std_real_box):
    """Standard Geometry"""
    coord = 1  # RZ
    periodicity = [0, 0, 1]
    gm = amr.Geometry(std_box, std_real_box, coord, periodicity)
    return gm


@pytest.fixture(scope="function")
def boxarr(std_box):
    """BoxArray for MultiFab creation"""
    ba = amr.BoxArray(std_box)
    ba.max_size(32)
    return ba


@pytest.fixture(scope="function")
def distmap(boxarr):
    """DistributionMapping for MultiFab creation"""
    dm = amr.DistributionMapping(boxarr)
    return dm


@pytest.fixture(scope="function", params=list(itertools.product([1, 3], [0, 1])))
def mfab(boxarr, distmap, request):
    """MultiFab that is either managed or device:
    The MultiFab object itself is not a fixture because we want to avoid caching
    it between amr.initialize/finalize calls of various tests.
    https://github.com/pytest-dev/pytest/discussions/10387
    https://github.com/pytest-dev/pytest/issues/5642#issuecomment-1279612764
    """

    class MfabContextManager:
        def __enter__(self):
            num_components = request.param[0]
            num_ghost = request.param[1]
            self.mfab = amr.MultiFab(boxarr, distmap, num_components, num_ghost)
            self.mfab.set_val(0.0, 0, num_components)
            return self.mfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.mfab.clear()
            del self.mfab

    with MfabContextManager() as mfab:
        yield mfab


@pytest.fixture(scope="function", params=list(itertools.product([1, 3], [0, 1])))
def mfab_device(boxarr, distmap, request):
    """MultiFab that resides purely on the device:
    The MultiFab object itself is not a fixture because we want to avoid caching
    it between amr.initialize/finalize calls of various tests.
    https://github.com/pytest-dev/pytest/discussions/10387
    https://github.com/pytest-dev/pytest/issues/5642#issuecomment-1279612764
    """

    class MfabDeviceContextManager:
        def __enter__(self):
            num_components = request.param[0]
            num_ghost = request.param[1]
            self.mfab = amr.MultiFab(
                boxarr,
                distmap,
                num_components,
                num_ghost,
                amr.MFInfo().set_arena(amr.The_Device_Arena()),
            )
            self.mfab.set_val(0.0, 0, num_components)
            return self.mfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.mfab.clear()
            del self.mfab

    with MfabDeviceContextManager() as mfab:
        yield mfab


@pytest.fixture(scope="function", params=list(itertools.product([1, 3], [0, 1])))
def imfab(boxarr, distmap, request):
    class iMfabContextManager:
        def __enter__(self):
            num_components = request.param[0]
            num_ghost = request.param[1]
            self.imfab = amr.iMultiFab(boxarr, distmap, num_components, num_ghost)
            self.imfab.set_val(0, 0, num_components)
            return self.imfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.imfab.clear()
            del self.imfab

    with iMfabContextManager() as imfab:
        yield imfab


@pytest.fixture(scope="function", params=list(itertools.product([1, 3], [0, 1])))
def imfab_device(boxarr, distmap, request):
    class iMfabDeviceContextManager:
        def __enter__(self):
            num_components = request.param[0]
            num_ghost = request.param[1]
            self.imfab = amr.iMultiFab(
                boxarr,
                distmap,
                num_components,
                num_ghost,
                amr.MFInfo().set_arena(amr.The_Device_Arena()),
            )
            self.imfab.set_val(0, 0, num_components)
            return self.imfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.imfab.clear()
            del self.imfab

    with iMfabDeviceContextManager() as imfab:
        yield imfab
