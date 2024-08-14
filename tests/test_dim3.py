# -*- coding: utf-8 -*-

import amrex.space3d as amr
import numpy as np


def test_dim3():
    obj = amr.Dim3(1, 2, 3)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3


def test_xdim3():
    obj = amr.XDim3(1.0, 2.0, 3.0)
    np.testing.assert_allclose(obj.x, 1.0)
    np.testing.assert_allclose(obj.y, 2.0)
    np.testing.assert_allclose(obj.z, 3.0)
