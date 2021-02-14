# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pyamrex as amrex

def test_dim3():
    obj = amrex.Dim3(1, 2, 3)
    assert(obj.x == 1)
    assert(obj.y == 2)
    assert(obj.z == 3)

def test_xdim3():
    obj = amrex.XDim3(1.0, 2.0, 3.0)
    np.testing.assert_allclose(obj.x, 1.0)
    np.testing.assert_allclose(obj.y, 2.0)
    np.testing.assert_allclose(obj.z, 3.0)
