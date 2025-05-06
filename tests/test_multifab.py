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

    assert iter(mfab).is_valid
    assert iter(mfab).length == 8

    cnt = 0
    for _mfi in mfab:
        cnt += 1

    assert iter(mfab).length == cnt


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
