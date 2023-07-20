# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


@pytest.mark.skipif(amr.Config.spacedim != 1, reason="Requires AMREX_SPACEDIM = 1")
def test_iv_1d():
    obj = amr.IntVect(1)
    assert obj[0] == 1
    assert obj[-1] == 1
    with pytest.raises(IndexError):
        obj[-2]
    with pytest.raises(IndexError):
        obj[1]


@pytest.mark.skipif(amr.Config.spacedim != 2, reason="Requires AMREX_SPACEDIM = 2")
def test_iv_2d():
    obj = amr.IntVect(1, 2)
    assert obj[0] == 1
    assert obj[1] == 2
    assert obj[-1] == 3
    assert obj[-2] == 2

    with pytest.raises(IndexError):
        obj[-3]
    with pytest.raises(IndexError):
        obj[2]


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_iv_3d1():
    obj = amr.IntVect(1, 2, 3)

    # Check indexing
    assert obj[0] == 1
    assert obj[1] == 2
    assert obj[2] == 3
    assert obj[-1] == 3
    assert obj[-2] == 2
    assert obj[-3] == 1
    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

    # Check properties
    assert obj.max == 3
    assert obj.min == 1
    assert obj.sum == 6

    # Check assignment
    obj[0] = 2
    obj[1] = 3
    obj[2] = 4
    assert obj[0] == 2
    assert obj[1] == 3
    assert obj[2] == 4


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_iv_3d2():
    obj = amr.IntVect(3)
    assert obj[0] == 3
    assert obj[1] == 3
    assert obj[2] == 3
    assert obj[-1] == 3
    assert obj[-2] == 3
    assert obj[-3] == 3

    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

    obj = amr.IntVect([2, 3, 4])
    assert obj[0] == 2
    assert obj[1] == 3
    assert obj[2] == 4


def test_iv_static():
    zero = amr.IntVect.zero_vector()
    for i in range(amr.Config.spacedim):
        assert zero[i] == 0

    one = amr.IntVect.unit_vector()
    for i in range(amr.Config.spacedim):
        assert one[i] == 1

    assert zero == amr.IntVect.cell_vector()
    assert one == amr.IntVect.node_vector()


def test_iv_ops():
    gold = amr.IntVect(2)
    one = amr.IntVect.unit_vector()

    two = one + one
    assert two == gold
    assert two != amr.IntVect.zero_vector()
    assert two > one
    assert two >= gold
    assert one < two
    assert one <= one

    assert not (one > two)

    zero = one - one
    assert zero == amr.IntVect.zero_vector()

    mtwo = one * gold
    assert two == mtwo

    four = amr.IntVect(4)
    dtwo = four / gold
    assert dtwo == mtwo


def test_iv_conversions():
    obj = amr.IntVect.max_vector().numpy()
    assert isinstance(obj, np.ndarray)
    assert obj.dtype == np.int32

    # check that memory is not collected too early
    iv = amr.IntVect(2)
    obj = iv.numpy()
    del iv
    assert obj[0] == 2


def test_iv_iter():
    a0 = amr.IntVect(4)
    b0 = amr.IntVect(2)

    a1 = [x // 2 for x in a0]
    b1 = [x for x in b0]

    np.testing.assert_allclose(a1, b1)


def test_iv_d_decl():
    iv = amr.IntVect(*amr.d_decl(1, 2, 3))
    assert iv == amr.IntVect(1, 2, 3)
