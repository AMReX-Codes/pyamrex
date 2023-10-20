# -*- coding: utf-8 -*-

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
