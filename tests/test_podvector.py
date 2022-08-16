# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex


def test_podvector_init():
    podv = amrex.PODVector_real()
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
    podv = amrex.PODVector_int()
    podv.push_back(1)
    podv.push_back(2)
    podv.push_back(1)
    podv.push_back(5)
    arr = np.array(podv, copy=False)
    print(arr)

    # podv[2] = 3
    arr[2] = 3
    print(arr)
    print(podv)
    assert arr[2] == podv[2] == 3

    podv[1] = 5
    assert arr[1] == podv[1] == 5
