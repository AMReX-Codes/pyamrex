# -*- coding: utf-8 -*-

import math

import numpy as np
import pytest

import amrex.space3d as amr


def test_imfab_numpy(imfab):
    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # imfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        # grow (aka guard/ghost/halo) regions
        ngv = imfab.n_grow_vect

        # get every local block of the field
        for mfi in imfab:
            # global index box w/ guards
            bx = mfi.tilebox().grow(ngv)
            print(bx)

            # numpy/cupy representation: non-
            # copying view, w/ guard/ghost
            field = imfab.array(mfi).to_xp()

            # notes on indexing in field:
            # - numpy uses locally zero-based indexing
            # - layout is F_CONTIGUOUS by default, just like AMReX

            field[()] = 42

    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # imfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        field_list = imfab.to_xp()

        for field in field_list:
            field[()] = 42

    # finest active MR level, get from a
    # simulation's AmrMesh object, e.g.:
    # finest_level = sim.finest_level
    finest_level = 0  # no MR

    # iterate over mesh-refinement levels
    for lev in range(finest_level + 1):
        # get an existing MultiFab, e.g.,
        # from a simulation:
        # imfab = sim.get_field(lev=lev)
        # Config = sim.extension.Config

        # Using global indexing
        # Set all valid cells (and internal ghost cells)
        imfab[...] = 42

        # Set a range of cells. Indices are in Fortran order.
        # First dimension, sets from first lower guard cell to first upper guard cell.
        #  - Imaginary indices refer to the guard cells, negative lower, positive upper.
        # Second dimension, sets all valid cells.
        # Third dimension, sets all valid and ghost cells
        #  - The empty tuple is used to specify the range to include all valid and ghost cells.
        # Components dimension, sets first component.
        imfab[-1j:2j, :, (), 0] = 42

        # Get a range of cells
        # Get the data along the valid cells in the first dimension (gathering data across blocks
        # and processors), at the first upper guard cell in the second dimensionn, and cell 2 of
        # the third (with 2 being relative to 0 which is the lower end of the valid cells of the full domain).
        # Note that in an MPI context, this is a global operation, so caution is required when
        # scaling to large numbers of processors.
        if imfab.n_grow_vect.max > 0:
            mfslice = imfab[:, 1j, 2]
            # The assignment is to the last valid cell of the second dimension.
            imfab[:, -1, 2] = 2 * mfslice


@pytest.mark.skipif(amr.Config.have_gpu, reason="This test only runs on CPU")
def test_imfab_loop_slow(imfab):
    ngv = imfab.n_grow_vect
    print(f"\n  imfab={imfab}, imfab.n_grow_vect={ngv}")

    for mfi in imfab:
        bx = mfi.tilebox().grow(ngv)
        marr = imfab.array(mfi)

        # print(imfab.num_comp)
        # print(imfab.size)
        # print(marr.size)
        # print(marr.nComp)

        # index by index assignment
        # notes:
        # - this is AMReX Array4, F-order indices
        # - even though we iterate by fastest varying index,
        #   such loops are naturally very slow in Python
        three_comps = imfab.num_comp == 3
        if three_comps:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k, 0] = 10 * i
                marr[i, j, k, 1] = 10 * j
                marr[i, j, k, 2] = 10 * k
        else:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k] = 10 * i

        # numpy representation: non-copying view, zero-indexed,
        # includes the guard/ghost region
        marr_np = marr.to_numpy()

        # check the values at start/end are the same: first component
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[-1, -1, -1, 0] == marr[bx.big_end]
        # same check, but for all components
        for n in range(imfab.num_comp):
            small_end_comp = list(bx.small_end) + [n]
            big_end_comp = list(bx.big_end) + [n]
            assert marr_np[0, 0, 0, n] == marr[small_end_comp]
            assert marr_np[-1, -1, -1, n] == marr[big_end_comp]

        # all components and all indices set at once to 42
        marr_np[()] = 42

        # values in start & end still match?
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[-1, -1, -1, -1] == marr[bx.big_end]

        # all values for all indices match between multifab & numpy view?
        for n in range(imfab.num_comp):
            for i, j, k in bx:
                assert marr[i, j, k, n] == 42


def test_imfab_loop(imfab):
    ngv = imfab.n_grow_vect
    print(f"\n  imfab={imfab}, imfab.n_grow_vect={ngv}")

    for mfi in imfab:
        bx = mfi.tilebox().grow(ngv)
        marr = imfab.array(mfi)

        # note: offset from index space in numpy
        #   in numpy, we start indices from zero, not small_end

        # numpy/cupy representation: non-copying view, including the
        # guard/ghost region
        marr_xp = marr.to_xp()

        marr_xp[()] = 10  # TODO: fill with index value or so as in test_imfab_loop_slow

        def iv2s(iv, comp):
            return tuple(iv) + (comp,)

        # check the values at start/end are the same: first component
        for n in range(imfab.num_comp):
            assert marr_xp[0, 0, 0, n] == 10
            assert marr_xp[-1, -1, -1, n] == marr_xp[iv2s(bx.big_end - bx.small_end, n)]

        # now we do some faster assignments, using range based access
        #   This should fail as out-of-bounds, but does not.
        #   Does NumPy/CuPy not check array access for non-owned views?
        # marr_xp[24:200, :, :, :] = 42.

        #   all components and all indices set at once to 42
        marr_xp[()] = 42

        # values in start & end still match?
        for n in range(imfab.num_comp):
            assert marr_xp[0, 0, 0, n] == 42
            assert marr_xp[-1, -1, -1, n] == marr_xp[iv2s(bx.big_end - bx.small_end, n)]


def test_imfab_simple(imfab):
    assert imfab.is_all_cell_centered
    # assert(all(not imfab.is_nodal(i) for i in [-1, 0, 1, 2]))  # -1??
    assert all(not imfab.is_nodal(i) for i in [0, 1, 2])

    for i in range(imfab.num_comp):
        imfab.set_val(-10 * (i + 1), i, 1)
    imfab.abs(0, imfab.num_comp)
    for i in range(imfab.num_comp):
        assert imfab.max(i) == (10 * (i + 1))  # Assert: None == 10 for i=0
        assert imfab.min(i) == (10 * (i + 1))

    imfab.plus(20, 0, imfab.num_comp)
    for i in range(imfab.num_comp):
        np.testing.assert_allclose(imfab.max(i), 20 + (10 * (i + 1)))
        np.testing.assert_allclose(imfab.min(i), 20 + (10 * (i + 1)))

    imfab.mult(10, 0, imfab.num_comp)
    for i in range(imfab.num_comp):
        np.testing.assert_allclose(imfab.max(i), 10 * (20 + (10 * (i + 1))))
        np.testing.assert_allclose(imfab.min(i), 10 * (20 + (10 * (i + 1))))

    imfab.negate(0, imfab.num_comp)
    for i in range(imfab.num_comp):
        np.testing.assert_allclose(imfab.max(i), -10 * (20 + (10 * (i + 1))))
        np.testing.assert_allclose(imfab.min(i), -10 * (20 + (10 * (i + 1))))


@pytest.mark.parametrize("nghost", [0, 1])
def test_imfab_ops(boxarr, distmap, nghost):
    src = amr.MultiFab(boxarr, distmap, 3, nghost)
    dst = amr.MultiFab(boxarr, distmap, 1, nghost)

    src.set_val(10, 0, 1)
    src.set_val(20, 1, 1)
    src.set_val(30, 2, 1)
    dst.set_val(0, 0, 1)

    dst.add(src, 2, 0, 1, nghost)
    dst.subtract(src, 1, 0, 1, nghost)
    dst.multiply(src, 0, 0, 1, nghost)
    dst.divide(src, 1, 0, 1, nghost)

    print(dst.min(0))
    np.testing.assert_allclose(dst.min(0), 5)
    np.testing.assert_allclose(dst.max(0), 5)


def test_imfab_mfiter(imfab):
    assert len(imfab) == 8

    assert iter(imfab).is_valid
    assert iter(imfab).length == 8

    cnt = 0
    for _mfi in imfab:
        cnt += 1

    assert iter(imfab).length == cnt


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_imfab_ops_cuda_numba(imfab_device):
    # https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    from numba import cuda

    ngv = imfab_device.n_grow_vect

    # assign 3: define kernel
    @cuda.jit
    def set_to_three(array):
        i, j, k = cuda.grid(3)
        if i < array.shape[0] and j < array.shape[1] and k < array.shape[2]:
            array[i, j, k] = 3

    # assign 3: loop through boxes and launch kernels
    for mfi in imfab_device:
        bx = mfi.tilebox().grow(ngv)  # noqa
        marr = imfab_device.array(mfi)
        marr_numba = cuda.as_cuda_array(marr)

        # kernel launch
        threadsperblock = (4, 4, 4)
        blockspergrid = tuple(
            [math.ceil(s / b) for s, b in zip(marr_numba.shape, threadsperblock)]
        )
        set_to_three[blockspergrid, threadsperblock](marr_numba)

    # Check results
    shape = 32**3 * 8
    sum_threes = imfab_device.sum_unique(comp=0, local=False)
    assert sum_threes == shape * 3


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_imfab_ops_cuda_cupy(imfab_device):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    import cupy as cp
    import cupyx.profiler

    # AMReX -> cupy
    ngv = imfab_device.n_grow_vect
    print(f"\n  imfab_device={imfab_device}, imfab_device.n_grow_vect={ngv}")

    # assign 3
    with cupyx.profiler.time_range("assign 3 [()]", color_id=0):
        for mfi in imfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = imfab_device.array(mfi).to_cupy(order="C")
            # print(marr_cupy.shape)  # 1, 32, 32, 32
            # print(marr_cupy.dtype)  # float64
            # performance:
            #   https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

            # write and read into the marr_cupy
            marr_cupy[()] = 3

    # verify result with a .sum_unique
    with cupyx.profiler.time_range("verify 3", color_id=0):
        shape = 32**3 * 8
        # print(imfab_device.shape)
        sum_threes = imfab_device.sum_unique(comp=0, local=False)
        assert sum_threes == shape * 3

    # assign 2
    with cupyx.profiler.time_range("assign 2 (set_val)", color_id=1):
        imfab_device.set_val(2)
    with cupyx.profiler.time_range("verify 2", color_id=1):
        sum_twos = imfab_device.sum_unique(comp=0, local=False)
        assert sum_twos == shape * 2

    # assign 5
    with cupyx.profiler.time_range("assign 5 (ones-like)", color_id=2):

        def set_to_five(mm):
            xp = cp.get_array_module(mm)
            assert xp.__name__ == "cupy"
            mm = xp.ones_like(mm) * 10
            mm /= 2
            return mm

        for mfi in imfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = imfab_device.array(mfi).to_cupy(order="F")
            # print(marr_cupy.shape)  # 32, 32, 32, 1
            # print(marr_cupy.dtype)  # float64
            # performance:
            #   https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

            # write and read into the marr_cupy
            fives_cp = set_to_five(marr_cupy)
            marr_cupy[()] = 0
            marr_cupy += fives_cp

    # verify
    with cupyx.profiler.time_range("verify 5", color_id=2):
        sum = imfab_device.sum_unique(comp=0, local=False)
        assert sum == shape * 5

    # assign 7
    with cupyx.profiler.time_range("assign 7 (fuse)", color_id=3):

        @cp.fuse(kernel_name="set_to_seven")
        def set_to_seven(x):
            x[...] = 7

        for mfi in imfab_device:
            bx = mfi.tilebox().grow(ngv)  # noqa
            marr_cupy = imfab_device.array(mfi).to_cupy(order="C")

            # write and read into the marr_cupy
            set_to_seven(marr_cupy)

    # verify
    with cupyx.profiler.time_range("verify 7", color_id=3):
        sum = imfab_device.sum_unique(comp=0, local=False)
        assert sum == shape * 7

    # TODO: @jit.rawkernel()


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_imfab_ops_cuda_pytorch(imfab_device):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
    import torch

    # assign 3: loop through boxes and launch kernel
    for mfi in imfab_device:
        marr = imfab_device.array(mfi)
        marr_torch = torch.as_tensor(marr, device="cuda")
        marr_torch[:, :, :] = 3

    # Check results
    shape = 32**3 * 8
    sum_threes = imfab_device.sum_unique(comp=0, local=False)
    assert sum_threes == shape * 3


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_imfab_ops_cuda_cuml(imfab_device):
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
def test_imfab_dtoh_copy(imfab_device):
    class MfabPinnedContextManager:
        def __enter__(self):
            self.imfab = amr.MultiFab(
                imfab_device.box_array(),
                imfab_device.dm(),
                imfab_device.n_comp,
                imfab_device.n_grow_vect,
                amr.MFInfo().set_arena(amr.The_Pinned_Arena()),
            )
            return self.imfab

        def __exit__(self, exc_type, exc_value, traceback):
            self.imfab.clear()
            del self.imfab

    with MfabPinnedContextManager() as imfab_host:
        imfab_host.set_val(42)

        amr.dtoh_memcpy(imfab_host, imfab_device)

        # assert all are 0 on host
        host_min = imfab_host.min(0)
        host_max = imfab_host.max(0)
        assert host_min == host_max
        assert host_max == 0

        dev_val = 11
        imfab_host.set_val(dev_val)
        amr.htod_memcpy(imfab_device, imfab_host)

        # assert all are 11 on device
        for n in range(imfab_device.n_comp):
            assert imfab_device.min(comp=n) == dev_val
            assert imfab_device.max(comp=n) == dev_val

        # numpy bindings (w/ copy)
        local_boxes_host = imfab_device.to_numpy(copy=True)
        assert max([np.max(box) for box in local_boxes_host]) == dev_val
        del local_boxes_host

        # numpy bindings (w/ copy)
        for mfi in imfab_device:
            marr = imfab_device.array(mfi).to_numpy(copy=True)
            assert np.min(marr) >= dev_val
            assert np.max(marr) <= dev_val

        # cupy bindings (w/o copy)
        import cupy as cp

        local_boxes_device = imfab_device.to_cupy()
        assert max([cp.max(box) for box in local_boxes_device]) == dev_val


def test_imfab_copy(imfab):
    # write to imfab
    imfab.set_val(42)
    for i in range(imfab.num_comp):
        np.testing.assert_allclose(imfab.max(i), 42)

    # copy
    new_imfab = imfab.copy()

    # write to old imfab
    imfab.set_val(1)
    for i in range(imfab.num_comp):
        np.testing.assert_allclose(imfab.max(i), 1)

    # check new imfab is the original data
    for i in range(new_imfab.num_comp):
        np.testing.assert_allclose(new_imfab.max(i), 42)
