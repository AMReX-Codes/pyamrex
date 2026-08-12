# -*- coding: utf-8 -*-

import math

import numpy as np
import pytest

import amrex.space3d as amr


def test_mfab_numpy(mfab):
    """Used in docs/source/usage/compute.rst"""

    # Manual: Compute Mfab Detailed START
    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # mfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        # grow (aka guard/ghost/halo) regions
        ngv = mfab.n_grow_vect

        # get every local block of the field
        for mfi in mfab:
            # global index box w/ guards
            bx = mfi.tilebox().grow(ngv)
            print(bx)

            # numpy/cupy representation: non-
            # copying view, w/ guard/ghost
            field = mfab.array(mfi).to_xp()

            # notes on indexing in field:
            # - numpy uses locally zero-based indexing
            # - layout is F_CONTIGUOUS by default, just like AMReX

            field[()] = 42.0
    # Manual: Compute Mfab Detailed END

    # Manual: Compute Mfab Simple START
    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # mfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        field_list = mfab.to_xp()

        for field in field_list:
            field[()] = 42.0
    # Manual: Compute Mfab Simple END

    # Manual: Compute Mfab Global START
    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # mfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        # Using global indexing
        # Set all cells, valid and ghost cells, using the empty tuple
        mfab[()] = np.pi

        # Set all valid cells (and ghost cells overlapping valid cells), using the Ellipsis
        mfab[...] = 42.0

        # Set a range of cells. Indices are in Fortran order.
        # First dimension, sets from first lower guard cell to first upper guard cell.
        #  - Imaginary indices refer to the guard cells, negative lower, positive upper.
        # Second dimension, sets all valid cells.
        # Third dimension, sets all valid and ghost cells
        #  - The empty tuple is used to specify the range to include all valid and ghost cells.
        # Components dimension, sets first component.
        mfab[-1j:2j, :, (), 0] = 37.0

        # Get a range of cells
        # Get the data along the valid cells in the first dimension (gathering data across blocks
        # and processors), at the first upper guard cell in the second dimensionn, and cell 2 of
        # the third (with 2 being relative to 0 which is the lower end of the valid cells of the full domain).
        # Note that in an MPI context, this is a global operation, so caution is required when
        # scaling to large numbers of processors.
        if mfab.n_grow_vect.max > 0:
            mfslice = mfab[:, 1j, 2]
            # The assignment is to the last valid cell of the second dimension.
            mfab[:, -1, 2] = 2 * mfslice

    # Manual: Compute Mfab Global END


@pytest.mark.skipif(amr.Config.have_gpu, reason="This test only runs on CPU")
def test_mfab_loop_slow(mfab):
    ngv = mfab.n_grow_vect
    print(f"\n  mfab={mfab}, mfab.n_grow_vect={ngv}")

    for mfi in mfab:
        bx = mfi.tilebox().grow(ngv)
        marr = mfab.array(mfi)

        # print(mfab)
        # print(mfab.num_comp)
        # print(mfab.size)
        # print(marr.size)
        # print(marr.nComp)

        # index by index assignment
        # notes:
        # - this is AMReX Array4, F-order indices
        # - even though we iterate by fastest varying index,
        #   such loops are naturally very slow in Python
        three_comps = mfab.num_comp == 3
        if three_comps:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k, 0] = 10.0 * i
                marr[i, j, k, 1] = 10.0 * j
                marr[i, j, k, 2] = 10.0 * k
        else:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k] = 10.0 * i

        # numpy representation: non-copying view, zero-indexed,
        # includes the guard/ghost region
        marr_np = marr.to_numpy()

        # check the values at start/end are the same: first component
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[-1, -1, -1, 0] == marr[bx.big_end]
        # same check, but for all components
        for n in range(mfab.num_comp):
            small_end_comp = list(bx.small_end) + [n]
            big_end_comp = list(bx.big_end) + [n]
            assert marr_np[0, 0, 0, n] == marr[small_end_comp]
            assert marr_np[-1, -1, -1, n] == marr[big_end_comp]

        # all components and all indices set at once to 42
        marr_np[()] = 42.0

        # values in start & end still match?
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[-1, -1, -1, -1] == marr[bx.big_end]

        # all values for all indices match between multifab & numpy view?
        for n in range(mfab.num_comp):
            for i, j, k in bx:
                assert marr[i, j, k, n] == 42.0


def test_mfab_loop(mfab):
    ngv = mfab.n_grow_vect
    print(f"\n  mfab={mfab}, mfab.n_grow_vect={ngv}")

    for mfi in mfab:
        bx = mfi.tilebox().grow(ngv)
        marr = mfab.array(mfi)

        # note: offset from index space in numpy
        #   in numpy, we start indices from zero, not small_end

        # numpy/cupy representation: non-copying view, including the
        # guard/ghost region
        marr_xp = marr.to_xp()

        marr_xp[()] = (
            10.0  # TODO: fill with index value or so as in test_mfab_loop_slow
        )

        def iv2s(iv, comp):
            return tuple(iv) + (comp,)

        # check the values at start/end are the same: first component
        for n in range(mfab.num_comp):
            assert marr_xp[0, 0, 0, n] == 10.0
            assert marr_xp[-1, -1, -1, n] == marr_xp[iv2s(bx.big_end - bx.small_end, n)]

        # now we do some faster assignments, using range based access
        #   This should fail as out-of-bounds, but does not.
        #   Does NumPy/CuPy not check array access for non-owned views?
        # marr_xp[24:200, :, :, :] = 42.

        #   all components and all indices set at once to 42
        marr_xp[()] = 42.0

        # values in start & end still match?
        for n in range(mfab.num_comp):
            assert marr_xp[0, 0, 0, n] == 42.0
            assert marr_xp[-1, -1, -1, n] == marr_xp[iv2s(bx.big_end - bx.small_end, n)]


def test_mfab_simple(mfab):
    assert mfab.is_all_cell_centered
    # assert(all(not mfab.is_nodal(i) for i in [-1, 0, 1, 2]))  # -1??
    assert all(not mfab.is_nodal(i) for i in [0, 1, 2])

    for i in range(mfab.num_comp):
        mfab.set_val(-10 * (i + 1), i, 1)
    mfab.abs(0, mfab.num_comp)
    for i in range(mfab.num_comp):
        assert mfab.max(i) == (10 * (i + 1))  # Assert: None == 10 for i=0
        assert mfab.min(i) == (10 * (i + 1))

    mfab.plus(20.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 20.0 + (10 * (i + 1)))
        np.testing.assert_allclose(mfab.min(i), 20.0 + (10 * (i + 1)))

    mfab.mult(10.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 10.0 * (20.0 + (10 * (i + 1))))
        np.testing.assert_allclose(mfab.min(i), 10.0 * (20.0 + (10 * (i + 1))))
    mfab.invert(10.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 1.0 / (20.0 + (10 * (i + 1))))
        np.testing.assert_allclose(mfab.min(i), 1.0 / (20.0 + (10 * (i + 1))))


@pytest.mark.parametrize("nghost", [0, 1])
def test_mfab_ops(boxarr, distmap, nghost):
    src = amr.MultiFab(boxarr, distmap, 3, nghost)
    dst = amr.MultiFab(boxarr, distmap, 1, nghost)

    src.set_val(10.0, 0, 1)
    src.set_val(20.0, 1, 1)
    src.set_val(30.0, 2, 1)
    dst.set_val(0.0, 0, 1)

    dst.add(src, 2, 0, 1, nghost)
    dst.subtract(src, 1, 0, 1, nghost)
    dst.multiply(src, 0, 0, 1, nghost)
    dst.divide(src, 1, 0, 1, nghost)

    print(dst.min(0))
    np.testing.assert_allclose(dst.min(0), 5.0)
    np.testing.assert_allclose(dst.max(0), 5.0)

    dst.xpay(2.0, src, 0, 0, 1, nghost)
    dst.saxpy(2.0, src, 1, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 60.0)
    np.testing.assert_allclose(dst.max(0), 60.0)

    dst.lin_comb(6.0, src, 1, 1.0, src, 2, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 150.0)
    np.testing.assert_allclose(dst.max(0), 150.0)


def test_mfab_mfiter(mfab):
    assert len(mfab) == 8

    # iter(mfab) is a generator driving an MFIter, not the MFIter itself, so
    # that leaving a loop early still finalizes it. Ask the MFIter directly.
    assert amr.MFIter(mfab).is_valid
    assert amr.MFIter(mfab).length == 8

    cnt = 0
    for _mfi in mfab:
        cnt += 1

    assert amr.MFIter(mfab).length == cnt


def test_mfab_mfiter_keeps_mfab_alive(mfab, assert_keeps_python_alive):
    assert_keeps_python_alive(mfab, lambda: iter(mfab))
    assert_keeps_python_alive(mfab, lambda: amr.MFIter(mfab))
    assert_keeps_python_alive(mfab, lambda: amr.MFIter(mfab, amr.MFItInfo()))


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_mfab_ops_cuda_numba(mfab_device):
    # https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    from numba import cuda

    ngv = mfab_device.n_grow_vect

    # assign 3: define kernel
    @cuda.jit
    def set_to_three(array):
        i, j, k = cuda.grid(3)
        if i < array.shape[0] and j < array.shape[1] and k < array.shape[2]:
            array[i, j, k] = 3.0

    # assign 3: loop through boxes and launch kernels
    for mfi in mfab_device:
        bx = mfi.tilebox().grow(ngv)  # noqa
        marr = mfab_device.array(mfi)
        marr_numba = cuda.as_cuda_array(marr)

        # kernel launch
        threadsperblock = (4, 4, 4)
        blockspergrid = tuple(
            [math.ceil(s / b) for s, b in zip(marr_numba.shape, threadsperblock)]
        )
        set_to_three[blockspergrid, threadsperblock](marr_numba)

    # Check results
    shape = 32**3 * 8
    sum_threes = mfab_device.sum_unique(comp=0, local=False)
    assert sum_threes == shape * 3


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_mfab_ops_cuda_cupy(mfab_device):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    import cupy as cp
    import cupyx.profiler

    # AMReX -> cupy
    ngv = mfab_device.n_grow_vect
    print(f"\n  mfab_device={mfab_device}, mfab_device.n_grow_vect={ngv}")

    # assign 3
    with cupyx.profiler.time_range("assign 3 [()]", color_id=0):
        for mfi in mfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = mfab_device.array(mfi).to_cupy(order="C")
            # print(marr_cupy.shape)  # 1, 32, 32, 32
            # print(marr_cupy.dtype)  # float64
            # performance:
            #   https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

            # write and read into the marr_cupy
            marr_cupy[()] = 3.0

    # verify result with a .sum_unique
    with cupyx.profiler.time_range("verify 3", color_id=0):
        shape = 32**3 * 8
        # print(mfab_device.shape)
        sum_threes = mfab_device.sum_unique(comp=0, local=False)
        assert sum_threes == shape * 3

    # assign 2
    with cupyx.profiler.time_range("assign 2 (set_val)", color_id=1):
        mfab_device.set_val(2.0)
    with cupyx.profiler.time_range("verify 2", color_id=1):
        sum_twos = mfab_device.sum_unique(comp=0, local=False)
        assert sum_twos == shape * 2

    # assign 5
    with cupyx.profiler.time_range("assign 5 (ones-like)", color_id=2):

        def set_to_five(mm):
            xp = cp.get_array_module(mm)
            assert xp.__name__ == "cupy"
            mm = xp.ones_like(mm) * 10.0
            mm /= 2.0
            return mm

        for mfi in mfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = mfab_device.array(mfi).to_cupy(order="F")
            # print(marr_cupy.shape)  # 32, 32, 32, 1
            # print(marr_cupy.dtype)  # float64
            # performance:
            #   https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

            # write and read into the marr_cupy
            fives_cp = set_to_five(marr_cupy)
            marr_cupy[()] = 0.0
            marr_cupy += fives_cp

    # verify
    with cupyx.profiler.time_range("verify 5", color_id=2):
        sum = mfab_device.sum_unique(comp=0, local=False)
        assert sum == shape * 5

    # assign 7
    with cupyx.profiler.time_range("assign 7 (fuse)", color_id=3):

        @cp.fuse(kernel_name="set_to_seven")
        def set_to_seven(x):
            x[...] = 7.0

        for mfi in mfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = mfab_device.array(mfi).to_cupy(order="C")

            # write and read into the marr_cupy
            set_to_seven(marr_cupy)

    # verify
    with cupyx.profiler.time_range("verify 7", color_id=3):
        sum = mfab_device.sum_unique(comp=0, local=False)
        assert sum == shape * 7

    # TODO: @jit.rawkernel()


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_mfab_ops_cuda_pytorch(mfab_device):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
    import torch

    # assign 3: loop through boxes and launch kernel
    for mfi in mfab_device:
        marr = mfab_device.array(mfi)
        marr_torch = torch.as_tensor(marr, device="cuda")
        marr_torch[:, :, :] = 3

    # Check results
    shape = 32**3 * 8
    sum_threes = mfab_device.sum_unique(comp=0, local=False)
    assert sum_threes == shape * 3


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_mfab_ops_cuda_cuml(mfab_device):
    pass
    # https://github.com/rapidsai/cuml
    # https://github.com/rapidsai/cudf
    #   maybe better for particles as a dataframe test
    # import cudf
    # import cuml

    # AMReX -> RAPIDSAI cuML
    # arr_cuml = ...
    # assert(arr_cuml.__cuda_array_interface__['data'][0] == arr.__cuda_array_interface__['data'][0])
    # TODO


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_mfab_dtoh_copy(mfab_device):
    class MfabPinnedContextManager:
        def __enter__(self):
            self.mfab = amr.MultiFab(
                mfab_device.box_array(),
                mfab_device.dm(),
                mfab_device.n_comp,
                mfab_device.n_grow_vect,
                amr.MFInfo().set_arena(amr.The_Pinned_Arena()),
            )
            return self.mfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.mfab.clear()
            del self.mfab

    with MfabPinnedContextManager() as mfab_host:
        mfab_host.set_val(42.0)

        amr.dtoh_memcpy(mfab_host, mfab_device)

        # assert all are 0.0 on host
        host_min = mfab_host.min(0)
        host_max = mfab_host.max(0)
        assert host_min == host_max
        assert host_max == 0.0

        dev_val = 11.0
        mfab_host.set_val(dev_val)
        amr.htod_memcpy(mfab_device, mfab_host)

        # assert all are 11.0 on device
        for n in range(mfab_device.n_comp):
            assert mfab_device.min(comp=n) == dev_val
            assert mfab_device.max(comp=n) == dev_val

        # numpy bindings (w/ copy)
        local_boxes_host = mfab_device.to_numpy(copy=True)
        assert max([np.max(box) for box in local_boxes_host]) == dev_val
        del local_boxes_host

        # numpy bindings (w/ copy)
        for mfi in mfab_device:
            marr = mfab_device.array(mfi).to_numpy(copy=True)
            assert np.min(marr) >= dev_val
            assert np.max(marr) <= dev_val

        # cupy bindings (w/o copy)
        import cupy as cp

        local_boxes_device = mfab_device.to_cupy()
        assert max([cp.max(box) for box in local_boxes_device]) == dev_val


def test_mfab_copy(mfab):
    # write to mfab
    mfab.set_val(42.0)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 42.0)

    # copy
    new_mfab = mfab.copy()

    # write to old mfab
    mfab.set_val(1.0)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 1.0)

    # check new mfab is the original data
    for i in range(new_mfab.num_comp):
        np.testing.assert_allclose(new_mfab.max(i), 42.0)


def test_mfab_mfiter_early_exit(mfab):
    """Leaving an MFIter loop early must still finalize the iterator.

    Otherwise MFIter::depth stays at 1 and the *next* iterator construction
    trips AMREX_ALWAYS_ASSERT(depth == 1), aborting the process. On GPU the
    Gpu::streamSynchronize() in MFIter::Finalize() would also be skipped.
    """

    def count():
        return sum(1 for _mfi in mfab)

    n = count()

    for _mfi in mfab:
        break
    assert count() == n

    for _mfi in mfab.tiles(tile=(16, 16, 16)):
        break
    assert count() == n

    for _mfi in amr.MFIter(mfab, amr.TilingIfNotGPU((16, 16, 16))):
        break
    assert count() == n

    def raises():
        for _mfi in mfab:
            raise ValueError("kernel error")

    with pytest.raises(ValueError):
        raises()
    assert count() == n

    # Finalize() is idempotent, so calling it by hand stays safe
    for mfi in mfab:
        mfi.finalize()
        break
    assert count() == n


def test_mfab_tiles(mfab):
    nboxes = sum(1 for _mfi in mfab)

    # no tile size given -> whole boxes, same as plain iteration
    assert sum(1 for _mfi in mfab.tiles()) == nboxes

    tiled = [mfi.tilebox() for mfi in mfab.tiles(tile=(16, 16, 16))]
    if amr.Config.have_gpu:
        # TilingIfNotGPU: never tile on GPU, where whole boxes are wanted
        assert len(tiled) == nboxes
    else:
        # a tile smaller than the 32^3 boxes subdivides them
        assert len(tiled) == 8 * nboxes

    # either way the tiles cover this rank's valid region exactly once
    local_pts = sum(mfi.validbox().num_pts for mfi in mfab)
    assert sum(bx.num_pts for bx in tiled) == local_pts

    # ix_type mirrors C++ mf.ixType().toIntVect()
    assert mfab.ix_type == mfab.box_array().ix_type().to_IntVect()


def assert_xp_equal(actual, desired):
    """Array equality that works on NumPy, CuPy and dpnp alike.

    numpy.testing on a device array raises "Implicit conversion to a NumPy
    array is not allowed", so compare on the device and coerce only the
    resulting scalar.
    """
    assert actual.shape == desired.shape
    assert bool((actual == desired).all())


def test_mfab_array4_call(mfab):
    """Array4.__call__ is a global-indexed, Box-shaped view."""
    ng = mfab.n_grow_vect

    for mfi in mfab:
        bx = mfi.tilebox()
        arr = mfab.array(mfi)

        view = arr(bx)
        # exactly AMREX_SPACEDIM dimensions: an Array4 is always 4D (i,j,k,n)
        # with extent-1 padding, and the unused axes must be dropped rather
        # than sliced, or `comp` would land on k in a 1D/2D build
        assert view.ndim == amr.Config.spacedim
        assert view.shape == tuple(bx.size)

        # a view, not a copy: a write through it is visible in a fresh view.
        # Checked behaviorally rather than via .base, which dpnp arrays do
        # not have.
        view[...] = 7.0
        assert bool((arr(bx) == 7.0).all())

        # matches a hand-sliced reference against the fab's global lower bound
        ref = arr.to_xp(copy=False, order="F")
        lo = amr.lbound(arr)
        lo = (lo.x, lo.y, lo.z)
        expected = ref[
            tuple(
                slice(bx.small_end[d] - lo[d], bx.big_end[d] - lo[d] + 1)
                for d in range(amr.Config.spacedim)
            )
            + (0,)
        ]
        assert_xp_equal(view, expected)

        # component selection
        for comp in range(mfab.num_comp):
            arr(bx, comp=comp)[...] = float(comp)
        for comp in range(mfab.num_comp):
            np.testing.assert_allclose(float(arr(bx, comp=comp).min()), float(comp))
            np.testing.assert_allclose(float(arr(bx, comp=comp).max()), float(comp))

        # shifted access reaches the guard cells
        if ng.max > 0:
            shifted = arr(bx, di=-1)
            assert shifted.shape == view.shape
            assert_xp_equal(
                shifted,
                ref[
                    tuple(
                        slice(
                            bx.small_end[d] + (-1 if d == 0 else 0) - lo[d],
                            bx.big_end[d] + (-1 if d == 0 else 0) - lo[d] + 1,
                        )
                        for d in range(amr.Config.spacedim)
                    )
                    + (0,)
                ],
            )
            del shifted

        # release the views: on a GPU build these are DLPack exports, and
        # amrex.finalize() refuses to run while any are still alive
        del view, ref, expected, arr
        break


def test_mfab_for_each_tile(mfab):
    """Serial, tiled and threaded runs must all agree."""

    def fill(tile, threads):
        mfab.set_val(0.0)

        @amr.for_each_tile(mfab, tile=tile, threads=threads)
        def _(bx, a):
            a(bx)[...] = 42.0

        return [mfab.min(c) for c in range(mfab.num_comp)] + [
            mfab.max(c) for c in range(mfab.num_comp)
        ]

    reference = fill(None, 1)
    assert reference[0] == 42.0
    assert fill((16, 16, 16), 1) == reference
    assert fill((16, 16, 16), 4) == reference
    assert fill(None, 4) == reference

    # the decorator returns the kernel unchanged
    @amr.for_each_tile(mfab)
    def my_kernel(bx, a):
        pass

    assert callable(my_kernel)
    assert my_kernel.__name__ == "my_kernel"


def test_mfab_portable_kernel(boxarr, distmap):
    """A ghost-cell stencil written once for CPU-serial, CPU-threaded and GPU.

    Ported from the 3D branch of ImpactX ForceFromSelfFields.cpp. The input
    fields are linear in the index coordinates, so the central differences are
    exact and can be checked against the analytic gradient.
    """
    dr = (0.1, 0.2, 0.4)
    coef = (1.0, 2.0, 3.0)

    phi = [amr.MultiFab(boxarr, distmap, 1, 1) for _ in range(3)]
    for mfi in phi[0]:
        gb = mfi.fabbox()
        idx = amr.xp.meshgrid(
            *[
                amr.xp.arange(gb.small_end[d], gb.big_end[d] + 1)
                for d in range(amr.Config.spacedim)
            ],
            indexing="ij",
        )
        for c in range(3):
            phi[c].array(mfi)(gb)[...] = coef[c] * idx[c]

    field = amr.MultiFab(boxarr, distmap, 1, 0)
    field.set_val(0.0)

    # Manual: Portable Kernel START
    # C++ equivalent, e.g. ImpactX ForceFromSelfFields.cpp:
    #
    #   #ifdef AMREX_USE_OMP
    #   #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #   #endif
    #   for (MFIter mfi(field, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    #   {
    #       Box const& bx = mfi.tilebox(field.ixType().toIntVect());
    #       Array4<Real> const& f  = field.array(mfi);
    #       Array4<Real> const& px = phi_x.array(mfi);
    #
    #       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    #       {
    #           f(i,j,k) = inv2dr[0] * (px(i-1,j,k) - px(i+1,j,k)) + ...;
    #       });
    #   }
    #
    # In Python the decorator stands in for the pragma and the MFIter line, its
    # parameter list for the tilebox and array(mfi) extractions, and its body
    # for the ParallelFor lambda. Index the Array4s with the Box: f(bx) is the
    # tile, px(bx, di=-1) is the same tile shifted by one cell in x.
    #
    # There is no OpenMP here: the Python loop is always serial, so on-node
    # parallelism is either MPI ranks, `threads=` below, or the GPU.
    inv2dr = [0.5 / d for d in dr]

    @amr.for_each_tile(field, *phi, tile=(16, 16, 16), threads=4)
    def _(bx, f, px, py, pz):
        f(bx)[...] = (
            inv2dr[0] * (px(bx, di=-1) - px(bx, di=+1))
            + inv2dr[1] * (py(bx, dj=-1) - py(bx, dj=+1))
            + inv2dr[2] * (pz(bx, dk=-1) - pz(bx, dk=+1))
        )

    # Manual: Portable Kernel END

    expected = -sum(coef[c] / dr[c] for c in range(3))
    np.testing.assert_allclose(field.min(0), expected)
    np.testing.assert_allclose(field.max(0), expected)

    # the untiled, unthreaded path must agree
    field.set_val(0.0)
    for mfi in field.tiles():
        bx = mfi.tilebox(field.ix_type)
        f = field.array(mfi)
        px, py, pz = (p.array(mfi) for p in phi)
        f(bx)[...] = (
            inv2dr[0] * (px(bx, di=-1) - px(bx, di=+1))
            + inv2dr[1] * (py(bx, dj=-1) - py(bx, dj=+1))
            + inv2dr[2] * (pz(bx, dk=-1) - pz(bx, dk=+1))
        )
    np.testing.assert_allclose(field.min(0), expected)
    np.testing.assert_allclose(field.max(0), expected)
