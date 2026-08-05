# -*- coding: utf-8 -*-

import ctypes

import numpy as np
import pytest

import amrex.space3d as amr


def pycapsule_is_valid(capsule, name):
    """Check a PyCapsule's name, without consuming it."""
    PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
    PyCapsule_IsValid.restype = ctypes.c_int
    PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return bool(PyCapsule_IsValid(capsule, name))


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
    print(f"arr: DLPack device info: {arr.__dlpack_device__()}")
    # print(f"arr: DLPack: {arr.__dlpack__()}")
    print(f"x.shape: {x.shape}")
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

    # copy to numpy using DLPack
    c_arr2np = np.from_dlpack(arr, copy=True)
    assert c_arr2np.ndim == 4
    assert c_arr2np.dtype == np.dtype("double")
    print(f"c_arr2np: {c_arr2np.__array_interface__}")
    np.testing.assert_array_equal(x, c_arr2np[0, :, :, :])
    assert c_arr2np[0, 1, 1, 1] == 42

    # view to numpy using DLPack
    v_arr2np = np.from_dlpack(arr)
    assert v_arr2np.ndim == 4
    assert v_arr2np.dtype == np.dtype("double")
    np.testing.assert_array_equal(x, v_arr2np[0, :, :, :])
    assert v_arr2np[0, 1, 1, 1] == 42

    # change original buffer once more
    x[1, 1, 1] = 43
    # the view reflects the change, the copy does not
    assert v_arr2np[0, 1, 1, 1] == 43
    assert c_arr2np[0, 1, 1, 1] == 42

    # write through the view
    v_arr2np[0, 0, 0, 0] = 7
    assert x[0, 0, 0] == 7
    x[0, 0, 0] = 1

    # copy array4 (view)
    c_arr = amr.Array4_double(arr)
    v_carr2np = np.from_dlpack(c_arr)
    x[1, 1, 1] = 44
    assert v_carr2np[0, 1, 1, 1] == 44


def test_array4_dlpack_capsules():
    x = np.ones((2, 3, 4))
    arr = amr.Array4_double(x)

    # legacy consumers (max_version=None) get an unversioned "dltensor"
    cap = arr.__dlpack__()
    assert pycapsule_is_valid(cap, b"dltensor")
    assert not pycapsule_is_valid(cap, b"dltensor_versioned")

    # a max_version below (1, 0) also selects the legacy capsule
    cap = arr.__dlpack__(max_version=(0, 8))
    assert pycapsule_is_valid(cap, b"dltensor")

    # DLPack 1.x consumers get a versioned capsule
    cap = arr.__dlpack__(max_version=(1, 1))
    assert pycapsule_is_valid(cap, b"dltensor_versioned")
    cap = arr.__dlpack__(max_version=(1, 0))
    assert pycapsule_is_valid(cap, b"dltensor_versioned")
    del cap

    # device info: host memory
    assert arr.__dlpack_device__() == (int(amr.DLDeviceType.kDLCPU), 0)

    # same-device request is fine, in tuple and enum form
    np.from_dlpack(arr)
    v = arr.__dlpack__(dl_device=(int(amr.DLDeviceType.kDLCPU), 0))
    assert pycapsule_is_valid(v, b"dltensor")
    v = arr.__dlpack__(dl_device=(amr.DLDeviceType.kDLCPU, 0))
    assert pycapsule_is_valid(v, b"dltensor")


def test_array4_dlpack_errors():
    x = np.ones((2, 3, 4))
    arr = amr.Array4_double(x)

    # stream is undefined for CPU tensors, on both the view and copy paths
    with pytest.raises(ValueError):
        arr.__dlpack__(stream=1)
    with pytest.raises(ValueError):
        arr.__dlpack__(stream=1, copy=True)
    with pytest.raises(ValueError):
        arr.__dlpack__(stream=2, copy=True)

    # transfers to non-CPU devices are unsupported
    with pytest.raises(BufferError):
        arr.__dlpack__(dl_device=(int(amr.DLDeviceType.kDLCUDA), 0))

    # malformed keyword arguments
    with pytest.raises(TypeError):
        arr.__dlpack__(max_version=1)
    with pytest.raises(TypeError):
        arr.__dlpack__(dl_device=(1,))
    with pytest.raises(TypeError):
        arr.__dlpack__(copy=1)


def test_array4_dlpack_keeps_alive(assert_keeps_python_alive):
    x = np.ones((2, 3, 4))
    arr = amr.Array4_double(x)

    # the consuming array holds a reference on the producing Array4 ...
    view = assert_keeps_python_alive(arr, lambda: np.from_dlpack(arr))
    # ... and mutations round-trip
    view[0, 0, 0, 0] = 3
    assert arr[0, 0, 0] == 3

    # a copy does not need to keep the producer alive, but stays valid
    copied = np.from_dlpack(arr, copy=True)
    arr = None
    view = None
    assert copied[0, 0, 0, 0] == 3


def test_array4_dlpack_empty():
    empty = amr.Array4_double()
    assert empty.__dlpack_device__() == (int(amr.DLDeviceType.kDLCPU), 0)
    v = np.from_dlpack(empty)
    assert v.size == 0


def test_array4_views_keep_sources_alive(assert_keeps_python_alive):
    x = np.ones((2, 3, 4))

    arr = assert_keeps_python_alive(x, lambda: amr.Array4_double(x))
    assert_keeps_python_alive(arr, lambda: amr.Array4_double(arr))
    assert_keeps_python_alive(arr, lambda: amr.Array4_double(arr, 0))
    assert_keeps_python_alive(arr, lambda: amr.Array4_double(arr, 0, 1))


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_dlpack_cupy(mfab_device):
    import cupy as cp

    # AMReX -> cupy: zero-copy view on the device
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        device_type, device_id = arr.__dlpack_device__()
        assert device_type == int(amr.DLDeviceType.kDLCUDA)
        assert device_id >= 0

        marr = cp.from_dlpack(arr)
        assert marr.dtype == cp.float64
        marr[...] = 5.0

    # mutations through the view are visible in the MultiFab
    for mfi in mfab_device:
        marr = cp.from_dlpack(mfab_device.array(mfi))
        assert cp.all(marr == 5.0)

    # device-side copy: isolated from the source
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        c_marr = arr.to_cupy(copy=True, order="C")
        c_marr[...] = 6.0
        assert cp.all(cp.from_dlpack(arr) == 5.0)
        break


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_dlpack_device_to_host(mfab_device):
    # np.from_dlpack(..., device="cpu") requests dl_device=(kDLCPU, 0),
    # which triggers a producer-side device-to-host copy
    mfab_device.set_val(7.0)
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        host = np.from_dlpack(arr, device="cpu")
        assert np.all(host == 7.0)

        # copy=False must refuse the device-to-host transfer
        with pytest.raises(BufferError):
            arr.__dlpack__(dl_device=(int(amr.DLDeviceType.kDLCPU), 0), copy=False)

        # a device-to-host copy hands over CPU memory, so a stream is invalid
        # (validated against the destination device, not the GPU source)
        with pytest.raises(ValueError):
            arr.__dlpack__(dl_device=(int(amr.DLDeviceType.kDLCPU), 0), stream=1)

        # DLPack only permits stream >= -1
        with pytest.raises(ValueError):
            arr.__dlpack__(stream=-2)
        # stream=-1 (no synchronization) is accepted on a device tensor view
        arr.__dlpack__(stream=-1)
        # a producer-made copy cannot run on a consumer stream: it requires
        # stream=None and rejects every other stream value
        arr.__dlpack__(copy=True)  # stream=None (default): OK
        for bad in (-1, 1, 2, 12345):
            with pytest.raises(BufferError):
                arr.__dlpack__(stream=bad, copy=True)
        break


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_dlpack_managed_device_id(boxarr, distmap):
    # DLPack requires device_id == 0 for managed memory (see dlpack.h)
    mf = amr.MultiFab(
        boxarr, distmap, 1, 0, amr.MFInfo().set_arena(amr.The_Managed_Arena())
    )
    mf.set_val(1.0)
    for mfi in mf:
        assert mf.array(mfi).__dlpack_device__() == (
            int(amr.DLDeviceType.kDLCUDAManaged),
            0,
        )
        break


@pytest.mark.skipif(
    amr.Config.gpu_backend != "CUDA", reason="Requires AMReX_GPU_BACKEND=CUDA"
)
def test_array4_dlpack_pytorch(mfab_device):
    import torch

    mfab_device.set_val(1.0)
    for mfi in mfab_device:
        arr = mfab_device.array(mfi)
        # zero-copy: torch passes its current stream for synchronization
        t = torch.from_dlpack(arr)
        assert t.is_cuda
        t += 1.0

    for mfi in mfab_device:
        t = torch.from_dlpack(mfab_device.array(mfi))
        assert torch.all(t == 2.0)


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
