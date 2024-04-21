# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


@pytest.fixture(scope="function")
def box():
    return amr.Box((0, 0, 0), (127, 127, 127))


def test_box_setget(box):
    print(box.length())
    print(box.numPts())
    assert box.numPts() == 128 * 128 * 128

    box.small_end = amr.IntVect(3)
    assert box.numPts() == 125 * 125 * 125

    box.big_end = amr.IntVect(40)
    assert box.numPts() == 38 * 38 * 38


def test_length(box):
    print(box.length())
    assert box.length() == amr.IntVect(128, 128, 128)

    domain = box.length()
    ncells = 1
    for ii in range(amr.Config.spacedim):
        ncells *= domain[ii]
    print("ncells by hand", ncells)
    print("ncells from box", box.numPts())
    assert ncells == box.numPts()


def test_num_pts(box):
    np.testing.assert_allclose(box.lo_vect, [0, 0, 0])
    np.testing.assert_allclose(box.hi_vect, [127, 127, 127])
    assert box.num_pts == 2**21
    assert box.volume == 2**21


def test_grow(box):
    """box.grow"""
    bx = box.grow(3)
    np.testing.assert_allclose(bx.lo_vect, [-3, -3, -3])
    np.testing.assert_allclose(bx.hi_vect, [130, 130, 130])
    assert bx.num_pts == (134**3)


def test_slab(box):
    """box.make_slab"""
    box.make_slab(direction=2, slab_index=60)
    np.testing.assert_allclose(box.lo_vect, [0, 0, 60])
    np.testing.assert_allclose(box.hi_vect, [127, 127, 60])
    assert box.num_pts == (128 * 128 * 1)


# def test_convert(box):
#    """Conversion to node"""
#    bx = box.convert(amr.CellIndex.NODE, amr.CellIndex.NODE, amr.CellIndex.NODE)
#    assert(bx.num_pts == 129**3)
#    assert(bx.volume == 128**3)

#    bx = box.convert(amr.CellIndex.NODE, amr.CellIndex.CELL, amr.CellIndex.CELL)
#    np.testing.assert_allclose(bx.hi_vect, [128, 127, 127])
#    assert(bx.num_pts == 129 * 128 * 128)
#    assert(bx.volume == 128**3)


@pytest.mark.parametrize("dir", [None, 0, 1, 2])
def test_surrounding_nodes(box, dir):
    """Surrounding nodes"""
    nx = np.array(box.hi_vect)

    if dir is None:
        bx = box.surrounding_nodes()
        assert bx.num_pts == 129**3
        assert bx.volume == 128**3
        nx += 1
        np.testing.assert_allclose(bx.hi_vect, nx)
    else:
        bx = box.surrounding_nodes(dir=dir)
        assert bx.num_pts == 129 * 128 * 128
        assert bx.volume == 128**3
        nx[dir] += 1
        np.testing.assert_allclose(bx.hi_vect, nx)


'''
@pytest.mark.parametrize("dir", [-1, 0, 1, 2])
def test_enclosed_cells(box, dir):
    """Enclosed cells"""
    bxn = box.convert(amr.CellIndex.NODE, amr.CellIndex.NODE, amr.CellIndex.NODE)
    nx = np.array(bxn.hi_vect)
    bx = bxn.enclosed_cells(dir)

    if dir < 0:
        assert(bx.num_pts == 128**3)
        assert(bx.volume == 128**3)
        nx -= 1
        np.testing.assert_allclose(bx.hi_vect, nx)
    else:
        assert(bx.num_pts == 129 * 129 * 128)
        assert(bx.volume == 128**3)
        nx[dir] -= 1
        np.testing.assert_allclose(bx.hi_vect, nx)
'''
