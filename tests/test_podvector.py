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

    # basic roundtrip
    arr = np.array([1.0, 2.5, 3.7, 4.0], dtype=np.float64)
    podv = amr.DeviceVector_real.from_numpy(arr)
    assert podv.size() == 4
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr)

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


def test_from_xp():
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0])
    podv = amr.DeviceVector_real.from_xp(arr)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr)


def test_from_xp_default():
    import numpy as np

    arr = np.array([5.0, 6.0, 7.0])
    podv = amr.DeviceVector_real.from_xp(arr)
    assert podv.size() == 3
    result = podv.to_numpy(copy=True)
    np.testing.assert_array_equal(result, arr)


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
