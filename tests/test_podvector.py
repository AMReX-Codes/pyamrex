# -*- coding: utf-8 -*-

import pytest

import amrex.space3d as amr


def test_podvector_init():
    podv = amr.PODVector_real_std()
    print(podv.__array_interface__)
    # podv[0] = 1
    # podv[2] = 3
    assert podv.size() == 0
    podv.push_back(1)
    podv.push_back(2)
    assert podv.size() == 2 and podv[1] == 2
    podv.pop_back()
    assert podv.size() == 1
    podv.push_back(2.14)
    assert not podv.empty()
    podv.push_back(3.1)
    podv[2] = 5
    assert podv.size() == 3 and podv[2] == 5
    podv.clear()
    assert podv.size() == 0
    assert podv.empty()


def test_array_interface():
    podv = amr.PODVector_int_std()
    podv.push_back(1)
    podv.push_back(2)
    podv.push_back(1)
    podv.push_back(5)
    arr = podv.to_numpy()
    print(arr)

    # podv[2] = 3
    arr[2] = 3
    print(arr)
    print(podv)
    assert arr[2] == podv[2] == 3

    podv[1] = 5
    assert arr[1] == podv[1] == 5


def test_from_numpy():
    import numpy as np

    # basic roundtrip (cast to the vector's element type so the test is
    # precision-agnostic, e.g. single-precision builds)
    arr = np.array([1.0, 2.5, 3.7, 4.0], dtype=np.float64)
    podv = amr.DeviceVector_real.from_numpy(arr)
    assert podv.size() == 4
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))

    # from_numpy creates a copy, not a view
    arr[0] = 999.0
    assert podv[0] != 999.0

    # empty array
    empty = np.array([], dtype=np.float64)
    podv_empty = amr.DeviceVector_real.from_numpy(empty)
    assert podv_empty.size() == 0

    # from list (array-like)
    podv_list = amr.DeviceVector_real.from_numpy([10.0, 20.0])
    assert podv_list.size() == 2
    assert podv_list[1] == 20.0


def test_from_numpy_normalizes_input():
    import numpy as np

    # non-contiguous (strided) input is made contiguous on the host
    base = np.array([1.0, 9.0, 2.0, 9.0, 3.0, 9.0], dtype=np.float64)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    podv = amr.DeviceVector_real.from_numpy(strided)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], result.dtype))

    # mismatched dtype is cast to the vector's element type
    ints = np.array([4, 5, 6], dtype=np.int32)
    podv2 = amr.DeviceVector_real.from_numpy(ints)
    result2 = podv2.to_numpy(copy=True)
    np.testing.assert_array_equal(result2, np.array([4.0, 5.0, 6.0], result2.dtype))


def test_to_device_empty():
    podv = amr.PODVector_int_std()
    device = podv.to_device()
    assert isinstance(device, amr.DeviceVector_int)
    assert device.size() == 0
    assert device.empty()


def test_to_device_from_host_vector():
    import numpy as np

    values = np.array([1, -2, 5, 8], dtype=np.int32)
    podv = amr.PODVector_int_std.from_numpy(values)
    device = podv.to_device()

    assert isinstance(device, amr.DeviceVector_int)
    assert device.size() == values.size
    result = device.to_numpy(copy=True)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    podv[0] = 99
    np.testing.assert_array_equal(
        device.to_numpy(copy=True), values.astype(result.dtype)
    )


def test_to_device_from_device_vector():
    import numpy as np

    values = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    podv = amr.DeviceVector_real.from_numpy(values)
    device = podv.to_device()

    assert isinstance(device, amr.DeviceVector_real)
    assert device.size() == values.size
    result = device.to_numpy(copy=True)
    np.testing.assert_array_equal(result, values.astype(result.dtype))

    podv[1] = 7.0
    np.testing.assert_array_equal(
        device.to_numpy(copy=True), values.astype(result.dtype)
    )


def test_podvector_dlpack():
    import numpy as np

    podv = amr.PODVector_int_std()
    for v in [1, 2, 1, 5]:
        podv.push_back(v)

    # host memory
    assert podv.__dlpack_device__() == (int(amr.DLDeviceType.kDLCPU), 0)

    # zero-copy view
    view = np.from_dlpack(podv)
    assert view.ndim == 1
    assert view.shape == (4,)
    view[2] = 3
    assert podv[2] == 3

    # isolated copy
    copied = np.from_dlpack(podv, copy=True)
    copied[0] = 42
    assert podv[0] == 1


def test_podvector_dlpack_keeps_alive(assert_keeps_python_alive):
    import numpy as np

    podv = amr.PODVector_real_std()
    podv.push_back(1.0)
    view = assert_keeps_python_alive(podv, lambda: np.from_dlpack(podv))
    view[0] = 2.0
    assert podv[0] == 2.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_device():
    cp = pytest.importorskip("cupy")
    import numpy as np

    values = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    podv = amr.NonManagedDeviceVector_real.from_numpy(values)

    device_type, _ = podv.__dlpack_device__()
    assert device_type in (
        int(amr.DLDeviceType.kDLCUDA),
        int(amr.DLDeviceType.kDLROCM),
        int(amr.DLDeviceType.kDLOneAPI),
    )

    # zero-copy view on the device
    marr = cp.from_dlpack(podv)
    cp.testing.assert_array_equal(marr, cp.asarray(values.astype(marr.dtype)))
    marr[0] = 7.0
    assert podv[0] == 7.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_pinned():
    import numpy as np

    # pinned memory is host-accessible: NumPy can view it directly
    pinned = amr.HostVector_real()
    pinned.push_back(1.0)
    pinned.push_back(2.0)

    view = np.from_dlpack(pinned)
    np.testing.assert_array_equal(view, np.array([1.0, 2.0], dtype=view.dtype))
    view[0] = 3.0
    assert pinned[0] == 3.0


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_podvector_dlpack_pinned_to_cupy():
    # pinned host memory is advertised as kDLCUDAHost/kDLROCMHost, which CuPy's
    # from_dlpack rejects; to_cupy() must stage a host-to-device copy instead
    cp = pytest.importorskip("cupy")
    import numpy as np

    pinned = amr.HostVector_real()
    for v in [1.0, 2.0, 3.0]:
        pinned.push_back(v)

    marr = pinned.to_cupy()
    cp.testing.assert_array_equal(marr, cp.asarray(np.array([1.0, 2.0, 3.0])))

    # the host-to-device staging copy must be synchronized before returning:
    # modifying the source afterwards must not change the (snapshot) result
    pinned[0] = 42.0
    cp.testing.assert_array_equal(marr, cp.asarray(np.array([1.0, 2.0, 3.0])))


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
@pytest.mark.parametrize(
    "ctor_name",
    ["NonManagedDeviceVector_real", "ManagedVector_real", "HostVector_real"],
)
def test_podvector_dlpack_empty_device_type(ctor_name):
    # an empty vector has a null data pointer; its DLPack device must be
    # classified from the allocator's arena kind so it does NOT change once
    # the vector holds data (previously an empty device/managed/pinned vector
    # was mislabeled kDLCPU and then flipped after the first push_back)
    gpu = (
        int(amr.DLDeviceType.kDLCUDA),
        int(amr.DLDeviceType.kDLROCM),
        int(amr.DLDeviceType.kDLOneAPI),
    )

    ctor = getattr(amr, ctor_name)
    empty = ctor()
    assert empty.size() == 0
    empty_dev = empty.__dlpack_device__()

    empty.push_back(1.0)
    assert empty.__dlpack_device__() == empty_dev, (
        "device type changed after allocation"
    )

    # a non-host-accessible allocator must report a GPU device even when empty
    if ctor_name == "NonManagedDeviceVector_real":
        assert empty_dev[0] in gpu


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_from_numpy_device_only():
    # device-only allocator: the host-to-device copy must work without CuPy
    import numpy as np

    arr = np.array([1.0, 2.5, 3.7, 4.0], dtype=np.float64)
    podv = amr.NonManagedDeviceVector_real.from_numpy(arr)
    assert podv.size() == 4
    # read back through an AMReX device-to-host copy (no CuPy either)
    result = podv.to_host().to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))


def test_from_xp():
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0])
    podv = amr.DeviceVector_real.from_xp(arr)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr.astype(result.dtype))


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_from_cp():
    cp = pytest.importorskip("cupy")

    arr = cp.array([1.0, 2.5, 3.7, 4.0], dtype=cp.float64)
    podv = amr.DeviceVector_real.from_cupy(arr)
    assert podv.size() == 4
    cp.testing.assert_array_equal(podv.to_cupy(), arr)

    arr[0] = 999.0
    assert podv[0] != 999.0

    empty = cp.array([], dtype=cp.float64)
    podv_empty = amr.DeviceVector_real.from_cupy(empty)
    assert podv_empty.size() == 0

    podv_list = amr.DeviceVector_real.from_cupy([10.0, 20.0])
    assert podv_list.size() == 2
    cp.testing.assert_array_equal(
        podv_list.to_cupy(),
        cp.array([10.0, 20.0], dtype=cp.float64),
    )


@pytest.mark.skipif(not amr.Config.have_gpu, reason="requires AMReX GPU support")
def test_host_from_cp():
    cp = pytest.importorskip("cupy")
    import numpy as np

    arr = cp.array([1.0, 2.5, 3.7, 4.0], dtype=cp.float64)
    podv = amr.HostVector_real.from_cupy(arr)
    assert podv.size() == 4
    np.testing.assert_array_equal(podv.to_numpy(), cp.asnumpy(arr))

    arr[0] = 999.0
    assert podv[0] != 999.0
