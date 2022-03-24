# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_array4_empty():
    empty = amrex.Array4_double()

    # Check properties
    assert(empty.size == 0)
    assert(empty.nComp == 0)

    # assign empty
    emptyc = amrex.Array4_double(empty)
    # Check properties
    assert(emptyc.size == 0)
    assert(emptyc.nComp == 0)

def test_array4():
    # from numpy (also a non-owning view)
    x = np.ones((2, 3, 4,))
    print(x.__array_interface__)
    arr = amrex.Array4_double(x)
    print(arr.__array_interface__)
    assert(arr.nComp == 1)

    x[1, 1, 1] = 42
    # TypeError: 'amrex.amrex_pybind.Array4_double' object is not subscriptable
    # assert(arr[1, 1, 1] == 42)

    # copy to numpy
    c_arr2np = np.array(arr, copy=True)  # segfaults on Windows
    assert(c_arr2np.ndim == 4)
    assert(c_arr2np.dtype == np.dtype("double"))
    np.testing.assert_array_equal(x, c_arr2np[:, :, :, 0])
    assert(c_arr2np[1, 1, 1] == 42)

    # view to numpy
    v_arr2np = np.array(arr, copy=False)
    assert(c_arr2np.ndim == 4)
    assert(v_arr2np.dtype == np.dtype("double"))
    np.testing.assert_array_equal(x, v_arr2np[:, :, :, 0])
    assert(v_arr2np[1, 1, 1] == 42)

    # change original buffer once more
    x[1, 1, 1] = 43
    assert(v_arr2np[1, 1, 1] == 43)

    # copy array4 (view)
    c_arr = amrex.Array4_double(arr)
    v_carr2np = np.array(c_arr, copy=False)
    x[1, 1, 1] = 44
    assert(v_carr2np[1, 1, 1] == 44)

    # from cupy

    # to numpy

    # to cupy

    return

    # Check indexing
    assert(obj[0] == 1)
    assert(obj[1] == 2)
    assert(obj[2] == 3)
    assert(obj[-1] == 3)
    assert(obj[-2] == 2)
    assert(obj[-3] == 1)
    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

    # Check assignment
    obj[0] = 2
    obj[1] = 3
    obj[2] = 4
    assert(obj[0] == 2)
    assert(obj[1] == 3)
    assert(obj[2] == 4)

#def test_iv_conversions():
#    obj = amrex.IntVect.max_vector().numpy()
#    assert(isinstance(obj, np.ndarray))
#    assert(obj.dtype == np.int32)
