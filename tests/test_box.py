# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

@pytest.fixture
def box():
    return amrex.Box((0, 0, 0), (127, 127, 127))

def test_length(box):
    print(box.length())
    assert(box.length() == amrex.IntVect(128,128,128))


    domain = box.length()
    ncells = 1
    for ii in range(amrex.Config.spacedim):
        ncells *= domain[ii]
    print('ncells by hand', ncells)
    print('ncells from box', box.numPts())
    assert(ncells == box.numPts())

def test_num_pts(box):
    np.testing.assert_allclose(box.lo_vect, [0, 0, 0])
    np.testing.assert_allclose(box.hi_vect, [127, 127, 127])
    assert(box.num_pts == 2**21)
    assert(box.volume == 2**21)

def test_grow(box):
    """box.grow"""
    bx = box.grow(3)
    np.testing.assert_allclose(bx.lo_vect, [-3, -3, -3])
    np.testing.assert_allclose(bx.hi_vect, [130, 130, 130])
    assert(bx.num_pts == (134**3))

#def test_convert(box):
#    """Conversion to node"""
#    bx = box.convert(amrex.CellIndex.NODE, amrex.CellIndex.NODE, amrex.CellIndex.NODE)
#    assert(bx.num_pts == 129**3)
#    assert(bx.volume == 128**3)

#    bx = box.convert(amrex.CellIndex.NODE, amrex.CellIndex.CELL, amrex.CellIndex.CELL)
#    np.testing.assert_allclose(bx.hi_vect, [128, 127, 127])
#    assert(bx.num_pts == 129 * 128 * 128)
#    assert(bx.volume == 128**3)


@pytest.mark.parametrize("dir", [None, 0, 1, 2])
def test_surrounding_nodes(box, dir):
    """Surrounding nodes"""
    nx = np.array(box.hi_vect)

    if dir is None:
        bx = box.surrounding_nodes()
        assert(bx.num_pts == 129**3)
        assert(bx.volume == 128**3)
        nx += 1
        np.testing.assert_allclose(bx.hi_vect, nx)
    else:
        bx = box.surrounding_nodes(dir=dir)
        assert(bx.num_pts == 129 * 128 * 128)
        assert(bx.volume == 128**3)
        nx[dir] += 1
        np.testing.assert_allclose(bx.hi_vect, nx)
'''
@pytest.mark.parametrize("dir", [-1, 0, 1, 2])
def test_enclosed_cells(box, dir):
    """Enclosed cells"""
    bxn = box.convert(amrex.CellIndex.NODE, amrex.CellIndex.NODE, amrex.CellIndex.NODE)
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
