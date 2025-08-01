# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def test_realvect_init():
    rv = amr.RealVect()
    rv2 = amr.RealVect([ii + 0.1 for ii in range(amr.Config.spacedim)])
    rv3 = amr.RealVect(0.3)

    for ii in range(amr.Config.spacedim):
        assert rv[ii] == 0
        assert np.isclose(rv2[ii], ii + 0.1)
        assert np.isclose(rv3[ii], 0.3)


@pytest.mark.skipif(amr.Config.spacedim != 1, reason="Requires AMREX_SPACEDIM = 1")
def test_rv_1d():
    obj = amr.RealVect(1.2)
    assert obj[0] == 1.2
    assert obj[-1] == 1.2
    with pytest.raises(IndexError):
        obj[-2]
    with pytest.raises(IndexError):
        obj[1]


@pytest.mark.skipif(amr.Config.spacedim != 2, reason="Requires AMREX_SPACEDIM = 2")
def test_rv_2d():
    obj = amr.RealVect(1.5, 2)
    assert obj[0] == 1.5
    assert obj[1] == 2
    assert obj[-1] == 2
    assert obj[-2] == 1.5

    with pytest.raises(IndexError):
        obj[-3]
    with pytest.raises(IndexError):
        obj[2]


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_rv_3d1():
    obj = amr.RealVect(1, 2, 3.14)

    # Check indexing
    assert obj[0] == 1
    assert obj[1] == 2
    assert np.isclose(obj[2], 3.14)
    assert np.isclose(obj[-1], 3.14)
    assert obj[-2] == 2
    assert obj[-3] == 1
    with pytest.raises(IndexError):
        obj[-4]
    with pytest.raises(IndexError):
        obj[3]

    # Check properties
    assert np.isclose(obj.sum, 6.14)

    # Check assignment
    obj[0] = 2.5
    obj[1] = -3
    obj[2] = 4
    assert obj[0] == 2.5
    assert obj[1] == -3
    assert obj[2] == 4


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_rv_3d2():
    # rv = amr.RealVect(0.5,0.4,0.2)
    rv2 = amr.RealVect(amr.IntVect(1, 2, 3))
    # assert(rv[1] == 0.4)
    assert rv2[2] == 3


def test_rv_static():
    zero = amr.RealVect.zero_vector()
    for i in range(amr.Config.spacedim):
        assert zero[i] == 0

    one = amr.RealVect.unit_vector()
    for i in range(amr.Config.spacedim):
        assert one[i] == 1


def test_comparison():
    uv = amr.RealVect.unit_vector()
    zv = amr.RealVect.zero_vector()
    zv2 = amr.RealVect()
    v1 = amr.RealVect([ii for ii in range(amr.Config.spacedim)])

    assert uv != zv
    assert zv == zv2
    assert uv > zv
    assert v1 >= zv
    assert not (v1 <= uv)
    assert not (uv >= v1)
    assert zv2 <= v1
    assert zv < uv


def test_unary():
    assert +amr.RealVect(1.5) == amr.RealVect(1.5)
    assert -amr.RealVect(1.5) == amr.RealVect(-1.5)


def test_addition():
    uv = amr.RealVect.unit_vector()
    zv = amr.RealVect.zero_vector()

    assert zv + 1 == uv
    zv2 = amr.RealVect()
    zv2 += 1.0
    assert zv2 == uv

    assert 1.0 + uv == amr.RealVect(2)
    assert uv + zv == uv


def test_subtraction():
    uv = amr.RealVect.unit_vector()
    # minus equal
    v3 = amr.RealVect([ii + 0.1 for ii in range(amr.Config.spacedim)])
    v4 = amr.RealVect([ii + 1.1 for ii in range(amr.Config.spacedim)])
    v5 = amr.RealVect([ii + 1.1 for ii in range(amr.Config.spacedim)])
    v6 = amr.RealVect(-1)
    v4 -= uv
    for ii in range(amr.Config.spacedim):
        assert 10 * int(v4[ii]) == 10 * int(v3[ii])
    v4 -= v6
    for ii in range(amr.Config.spacedim):
        assert v4[ii] == v5[ii]

    # v - v
    res1 = v5 - v3
    for ii in range(amr.Config.spacedim):
        assert np.isclose(res1[ii], uv[ii])
    # r - v
    res2 = 1.0 - v6
    for ii in range(amr.Config.spacedim):
        assert np.isclose(res2[ii], 2.0)


def test_multiplication():
    # times equal
    v1 = amr.RealVect(1.5)
    v1 *= 2
    for ii in range(amr.Config.spacedim):
        assert v1[ii] == 3
    # times equal
    v1 = amr.RealVect(1.5)
    v2 = amr.RealVect([ii for ii in range(amr.Config.spacedim)])
    v2 *= v1
    for ii in range(amr.Config.spacedim):
        assert v2[ii] == 1.5 * ii
    # times
    v1 = amr.RealVect(1.5)
    assert v1 * 3 == amr.RealVect(4.5)
    assert 3 * v1 == amr.RealVect(4.5)
    assert v1 * amr.RealVect(3) == amr.RealVect(4.5)
    assert v1 * amr.RealVect(2.0) == amr.RealVect(3.0)
    # scale
    assert v1.scale(3) == amr.RealVect(4.5)


def test_dot_cross():
    # dotProduct
    v1 = amr.RealVect(1.5)
    v2 = amr.RealVect([1 + ii for ii in range(amr.Config.spacedim)])
    dims = amr.Config.spacedim
    assert v1.dotProduct(v2) == 1.5 * int(dims * (dims + 1) / 2)

    # crossProduct
    if amr.Config.spacedim == 3:
        v1 = amr.RealVect(1.5)
        v2 = amr.RealVect([ii for ii in range(amr.Config.spacedim)])
        assert v1.crossProduct(v2) == amr.RealVect(1.5, -3, 1.5)


def test_divide():
    # divide equal (2)
    v1 = amr.RealVect(3.0)
    v1 /= 2.0
    assert v1 == amr.RealVect(1.5)

    v2 = amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    v2 /= amr.RealVect(3)
    assert v2 == amr.RealVect([ii for ii in range(amr.Config.spacedim)])
    # divide
    assert amr.RealVect(3.0) / amr.RealVect(2.0) == amr.RealVect(1.5)
    assert amr.RealVect(3.0) / 1.5 == amr.RealVect(2.0)
    assert 3.0 / amr.RealVect(1.5) == amr.RealVect(2.0)


def test_rounding():
    # floor
    vlower = amr.IntVect([1 + ii for ii in range(amr.Config.spacedim)])
    vupper = amr.IntVect([2 + ii for ii in range(amr.Config.spacedim)])
    v2 = amr.RealVect([1.5 + ii for ii in range(amr.Config.spacedim)])
    v3 = amr.RealVect([1.49 + ii for ii in range(amr.Config.spacedim)])
    assert v2.floor() == vlower
    assert v3.ceil() == vupper
    assert v2.round() == vupper
    assert v3.round() == vlower


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_3d_max():
    # max
    v1 = amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    v2 = amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])
    ref_list = [6, 3, 6]
    compare_vect = amr.RealVect([ref_list[ii] for ii in range(amr.Config.spacedim)])
    v3 = amr.max(v1, v2)
    assert v3 == compare_vect
    assert v1 == amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    assert v2 == amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    v1.max(v2)
    assert v1 == compare_vect
    assert v2 == amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    v1 = amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    v2 = amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    amr.RealVect.max(v2, v1)
    assert v2 == compare_vect
    assert v1 == amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_3d_min():
    # min
    v1 = amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    v2 = amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])
    ref_list = [0, 3, 0]
    compare_vect = amr.RealVect([ref_list[ii] for ii in range(amr.Config.spacedim)])
    v3 = amr.min(v1, v2)
    assert v3 == compare_vect
    assert v1 == amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    assert v2 == amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    v1.min(v2)
    assert v1 == compare_vect
    assert v2 == amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    v1 = amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])
    v2 = amr.RealVect([3 * (2 - ii) for ii in range(amr.Config.spacedim)])

    amr.RealVect.min(v2, v1)
    assert v2 == compare_vect
    assert v1 == amr.RealVect([3 * ii for ii in range(amr.Config.spacedim)])


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_properties():
    tl = [1, -5, 4.2]
    v1 = amr.RealVect(tl)

    # sum
    assert np.isclose(v1.sum, sum(tl))
    # vectorLength
    assert np.isclose(v1.vectorLength, np.linalg.norm(tl))
    # // radSquared
    assert np.isclose(v1.radSquared, np.linalg.norm(tl) ** 2)
    # // product
    assert np.isclose(v1.product, np.prod(tl))
    # // minDir
    assert v1.minDir(False) == 1
    assert v1.minDir(True) == 0
    # // maxDir
    assert v1.maxDir(False) == 2
    assert v1.maxDir(True) == 1


def test_basis_vector():
    for ii in range(amr.Config.spacedim):
        ei = amr.RealVect.BASISREALV(ii)
        for jj in range(amr.Config.spacedim):
            if ii == jj:
                assert ei[jj] == 1
            else:
                assert ei[jj] == 0
