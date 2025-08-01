# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr
from amrex.space3d import RealVect as RV
from amrex.space3d import XDim3


def test_realbox_empty():
    rb = amr.RealBox()
    assert not rb.ok()
    assert rb.xlo[0] == 0 and rb.xlo[1] == 0 and rb.xlo[2] == 0 and rb.xhi[0] == -1
    assert rb.length(2) == -1 and rb.length(2) == -1 and rb.length(2) == -1
    assert rb.volume() == 0

    rb.setLo([-1, -2, -3])
    rb.setHi([0, 0, 0])
    assert np.isclose(rb.volume(), 6)
    rb.setHi(0, 1.2)
    assert np.isclose(rb.hi(0), 1.2)


def test_realbox_frombox(std_box):
    dx = [1.0, 2.0, 3.0]
    base = [100.0, 200.0, 300.0]  # offset
    rb1 = amr.RealBox(std_box, dx, base)

    rb2 = amr.RealBox(
        [100.0, 200.0, 300.0],
        [100.0 + dx[0] * 64, 200.0 + dx[1] * 64, 300.0 + dx[2] * 64],
    )
    assert amr.AlmostEqual(rb1, rb2)


def test_realbox_contains():
    rb1 = amr.RealBox([0, 0, 0], [1, 2, 1.5])
    point1 = [-1, 0, 2]
    point2 = [0.5, 0.5, 0.5]
    assert not rb1.contains(point1)
    assert rb1.contains(point2)

    rb2 = amr.RealBox(0.1, 0.2, 0.3, 0.3, 1, 1.0)
    rb3 = amr.RealBox([4, 5.0, 6.0], [5.0, 5.5, 7.0])
    assert rb1.contains(rb2)
    assert not rb1.contains(rb3)
    tol = 8
    assert rb1.contains(rb3, eps=tol)

    # contains Real Vect
    rv1 = RV(point1)
    rv2 = RV(point2)
    assert not rb1.contains(rv1)
    assert rb1.contains(rv2)
    tol = 0.1
    assert not rb1.contains(rv1, eps=tol)
    tol = 2
    assert rb1.contains(rb1, eps=tol)
    # contains XDim3
    d1 = XDim3(-1, 0, 2)
    d2 = XDim3(0.5, 0.5, 0.5)
    assert not rb1.contains(d1)
    assert rb1.contains(d2)


def test_realbox_intersects():
    rb1 = amr.RealBox([0, 0, 0], [1, 2, 1.5])
    rb2 = amr.RealBox([0.1, 0.2, 0.3], [0.3, 1, 1.0])
    rb3 = amr.RealBox([4, 5.0, 6.0], [5.0, 5.5, 7.0])
    assert rb1.intersects(rb2)
    assert not rb3.intersects(rb1)


def test_almost_equal():
    rb1 = amr.RealBox([0, 0, 0], [1, 2, 1.5])
    rb2 = amr.RealBox([0, 0, 0], [1, 2, 1.5])
    rb3 = amr.RealBox([0, 0.1, -0.1], [1, 2, 1.4])
    assert amr.AlmostEqual(rb1, rb2)
    assert not amr.AlmostEqual(rb1, rb3)
    tol = 0.05
    assert not amr.AlmostEqual(rb1, rb3, tol)
    tol = 0.2
    assert amr.AlmostEqual(rb1, rb3, tol)
