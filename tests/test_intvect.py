# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pyamrex as amrex

@pytest.mark.skipif(amrex.Config.spacedim != 1,
                    reason="Requires AMREX_SPACEDIM = 1")
def test_iv_1d():
    obj = amrex.IntVect(1)
    assert(obj[0] == 1)
    assert(obj[-1] == 1)
    with pytest.raises(IndexError):
        obj[-2]
    with pytest.raises(IndexError):
        obj[1]

@pytest.mark.skipif(amrex.Config.spacedim != 2,
                    reason="Requires AMREX_SPACEDIM = 2")
def test_iv_2d():
    obj = amrex.IntVect(1, 2)
    assert(obj[0] == 1)
    assert(obj[1] == 2)
    assert(obj[-1] == 3)
    assert(obj[-2] == 2)

    with pytest.raises(IndexError):
        obj[-3]
    with pytest.raises(IndexError):
        obj[2]

@pytest.mark.skipif(amrex.Config.spacedim != 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_iv_3d1():
    obj = amrex.IntVect(1, 2, 3)

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

    # Check properties
    assert(obj.max == 3)
    assert(obj.min == 1)
    assert(obj.sum == 6)

    # Check assignment
    obj[0] = 2
    obj[1] = 3
    obj[2] = 4
    assert(obj[0] == 2)
    assert(obj[1] == 3)
    assert(obj[2] == 4)

@pytest.mark.skipif(amrex.Config.spacedim != 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_iv_3d2():
    obj = amrex.IntVect(3)
    assert(obj[0] == 3)
    assert(obj[1] == 3)
    assert(obj[2] == 3)
    assert(obj[-1] == 3)
    assert(obj[-2] == 3)
    assert(obj[-3] == 3)

    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

def test_iv_static():
    zero = amrex.IntVect.zero_vector()
    for i in range(amrex.Config.spacedim):
        assert(zero[i] == 0)

    one = amrex.IntVect.unit_vector()
    for i in range(amrex.Config.spacedim):
        assert(one[i] == 1)

    assert(zero == amrex.IntVect.cell_vector())
    assert(one == amrex.IntVect.node_vector())

def test_iv_ops():
    gold = amrex.IntVect(2)
    one = amrex.IntVect.unit_vector()

    two = one + one
    assert(two == gold)
    assert(two != amrex.IntVect.zero_vector())
    assert(two > one)
    assert(two >= gold)
    assert(one < two)
    assert(one <= one)

    assert(not (one > two))

    zero = one - one
    assert(zero == amrex.IntVect.zero_vector())

    mtwo = one * gold
    assert(two == mtwo)

    four = amrex.IntVect(4)
    dtwo = four / gold
    assert(dtwo == mtwo)

def test_iv_conversions():
    obj = amrex.IntVect.max_vector().numpy()
    assert(isinstance(obj, np.ndarray))
    assert(obj.dtype == np.int32)
