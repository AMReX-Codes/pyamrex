# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def test_array4_empty():
    empty = amr.Array4_double()

    # Check properties
    assert empty.size == 0
    assert empty.nComp == 0

    # assign empty
    emptyc = amr.Array4_double(empty)
    # Check properties
    assert emptyc.size == 0
    assert emptyc.nComp == 0


def test_array4():
    # from numpy (also a non-owning view)
    x = np.ones(
        (
            2,
            3,
            4,
        )
    )
    print(f"\nx: {x.__array_interface__} {x.dtype}")
    arr = amr.Array4_double(x)
    print(f"arr: {arr.__array_interface__}")
    print(arr)
    assert arr.nComp == 1

    # change original array
    x[1, 1, 1] = 42
    # check values in Array4 view changed
    assert arr[1, 1, 1] == 42
    assert arr[1, 1, 1, 0] == 42  # with component
    # check existing values stayed
    assert arr[0, 0, 0] == 1
    assert arr[3, 2, 1] == 1

    # copy to numpy
    c_arr2np = np.array(arr, copy=True)  # segfaults on Windows
    assert c_arr2np.ndim == 4
    assert c_arr2np.dtype == np.dtype("double")
    print(f"c_arr2np: {c_arr2np.__array_interface__}")
    np.testing.assert_array_equal(x, c_arr2np[0, :, :, :])
    assert c_arr2np[0, 1, 1, 1] == 42

    # view to numpy
    v_arr2np = np.array(arr, copy=False)
    assert c_arr2np.ndim == 4
    assert v_arr2np.dtype == np.dtype("double")
    np.testing.assert_array_equal(x, v_arr2np[0, :, :, :])
    assert v_arr2np[0, 1, 1, 1] == 42

    # change original buffer once more
    x[1, 1, 1] = 43
    assert v_arr2np[0, 1, 1, 1] == 43

    # copy array4 (view)
    c_arr = amr.Array4_double(arr)
    v_carr2np = np.array(c_arr, copy=False)
    x[1, 1, 1] = 44
    assert v_carr2np[0, 1, 1, 1] == 44


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_numba():
    # https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    from numba import cuda

    # numba -> AMReX Array4
    x = np.ones(
        (
            2,
            3,
            4,
        )
    )  # type: numpy.ndarray

    # host-to-device copy
    x_numba = cuda.to_device(x)  # noqa
    #   type is numba.cuda.cudadrv.devicearray.DeviceNDArray
    # x_cupy = cupy.asarray(x_numba)
    #   type is cupy.ndarray

    # TODO: Implement __cuda_array_interface__ or DLPack in Array4 constructor
    # x_arr = amr.Array4_double(x_numba)  # type: amr.Array4_double

    # assert (
    #     x_arr.__cuda_array_interface__["data"][0]
    #     == x_numba.__cuda_array_interface__["data"][0]
    # )


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_cupy():
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    import cupy as cp

    # cupy -> AMReX Array4
    x = np.ones(
        (
            2,
            3,
            4,
        )
    )  # TODO: merge into next line and create on device?
    x_cupy = cp.asarray(x)  # type: cupy.ndarray
    print(f"x_cupy={x_cupy}")
    print(x_cupy.__cuda_array_interface__)

    # TODO: Implement __cuda_array_interface__ or DLPack in Array4 constructor
    # cupy -> AMReX array4
    # x_arr = amr.Array4_double(x_cupy)  # type: amr.Array4_double
    # print(f"x_arr={x_arr}")
    # print(x_arr.__cuda_array_interface__)

    # assert (
    #     x_arr.__cuda_array_interface__["data"][0]
    #     == x_cupy.__cuda_array_interface__["data"][0]
    # )


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_pytorch():
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
    # arr_torch = torch.as_tensor(arr, device='cuda')
    # assert(arr_torch.__cuda_array_interface__['data'][0] == arr.__cuda_array_interface__['data'][0])
    # TODO

    pass
