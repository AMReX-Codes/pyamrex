# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_podvector_init():
    podv = amrex.PODVector_real()
    print(podv.__array_interface__)
    # podv[0] = 1
    # podv[2] = 3
    print(podv.size())
    assert(podv.size() == 0)
    podv.push_back(1)
    podv.push_back(2)
    print(podv.size())
    assert(podv.size() == 2 and podv[1] == 2)
    print(podv)
    podv.pop_back()
    assert(podv.size() == 1)
    print(podv)
    podv.push_back(2.14)
    print('Is PODVector empty?', podv.empty())
    assert(not podv.empty())
    podv.push_back(3.1)
    podv[2] = 5
    print(podv)
    assert(podv.size() == 3 and podv[2] == 5)
    podv.clear()
    assert(podv.size() == 0)
    print(podv)
    print('Is PODVector empty now?', podv.empty())
    assert(podv.empty())

def test_array_interface():
    podv = amrex.PODVector_int()
    podv.push_back(1)
    podv.push_back(2)
    podv.push_back(1)
    podv.push_back(5)
    myarr = np.array([1,2,1,5])
    arr = np.array(podv)
    print(arr)
    # print(myarr.__array_interface__)
    # pv = amrex.PODVector_int(myarr)
    # print(pv.__array_interface__)
