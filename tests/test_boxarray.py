# -*- coding: utf-8 -*-

import amrex.space3d as amr


def test_boxarray(std_box):
    ba = amr.BoxArray(std_box)

    assert ba.size == 1
    ba.max_size(32)
    assert ba.size == 8

    assert not ba.empty
    assert ba.numPts == 64**3


def test_boxarray_empty():
    ba = amr.BoxArray()

    assert ba.size == 0
    assert ba.empty
    assert ba.numPts == 0


def test_boxarray_list():
    bx_1 = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(31, 31, 31))
    bx_2 = amr.Box(amr.IntVect(32, 32, 32), amr.IntVect(63, 63, 63))
    bx_3 = amr.Box(amr.IntVect(64, 64, 64), amr.IntVect(95, 95, 95))

    box_list = amr.Vector_Box([bx_1, bx_2, bx_3])
    ba = amr.BoxArray(box_list)
    print(ba)

    assert ba.size == 3
    assert not ba.empty
    assert ba.numPts == 98304

    print(ba.d_numPts)
    print(ba.ix_type())
