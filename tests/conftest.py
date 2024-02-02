# -*- coding: utf-8 -*-

import itertools
import os

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


@pytest.fixture(autouse=True, scope="function")
def amrex_init(tmpdir):
    with tmpdir.as_cwd():
        amr.initialize(
            [
                # print AMReX status messages
                "amrex.verbose=2",
                # throw exceptions and create core dumps instead of
                # AMReX backtrace files: allows to attach to
                # debuggers
                "amrex.throw_exception=1",
                "amrex.signal_handling=0",
                # abort GPU runs if out-of-memory instead of swapping to host RAM
                # "abort_on_out_of_gpu_memory=1",
                # the arena for GPU runs defaults to managed
                "amrex.the_arena_is_managed=1",
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
