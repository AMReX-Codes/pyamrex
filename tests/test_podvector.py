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
