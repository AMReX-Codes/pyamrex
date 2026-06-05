# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def test_tagval_values():
    assert int(amr.TagVal.CLEAR) == 0
    assert int(amr.TagVal.BUF) == 1
    assert int(amr.TagVal.SET) == 2


def test_tagboxarray_construct(boxarr, distmap):
    tba = amr.TagBoxArray(boxarr, distmap, 1)
    assert tba.n_grow_vect == amr.IntVect(1)
    assert tba.box_array() == boxarr

    tba = amr.TagBoxArray(boxarr, distmap, amr.IntVect(2))
    assert tba.n_grow_vect == amr.IntVect(2)


def test_tagboxarray_setval(boxarr, distmap, std_box):
    tba = amr.TagBoxArray(boxarr, distmap)
    assert not tba.has_tags(std_box)

    tba.set_val(amr.TagVal.SET)
    assert tba.has_tags(std_box)

    tba.set_val(amr.TagVal.CLEAR)
    assert not tba.has_tags(std_box)

    # set tags in a sub-region only
    sub_box = amr.Box(amr.IntVect(8), amr.IntVect(15))
    other_box = amr.Box(amr.IntVect(24), amr.IntVect(31))
    tba.set_val(amr.BoxArray(sub_box), amr.TagVal.SET)
    assert tba.has_tags(sub_box)
    assert not tba.has_tags(other_box)


def test_tagboxarray_array_mask(boxarr, distmap):
    """Tag cells through zero-copy Array4 (of char) views, as a Python
    AmrCore.error_est override does"""
    sd = amr.Config.spacedim
    tba = amr.TagBoxArray(boxarr, distmap)

    tag_box = amr.Box(amr.IntVect(4), amr.IntVect(11))

    for mfi in tba:
        bx = mfi.validbox()
        tags = tba.array(mfi).to_xp(copy=False, order="F")
        assert tags.dtype == np.int8

        # global index arrays for the local view (no ghost cells here)
        idx = [
            np.arange(bx.small_end[d], bx.big_end[d] + 1).reshape(
                [-1 if i == d else 1 for i in range(4)]
            )
            for d in range(sd)
        ]
        inside = np.ones_like(tags, dtype=bool)
        for d in range(sd):
            inside &= (idx[d] >= tag_box.small_end[d]) & (idx[d] <= tag_box.big_end[d])
        tags[inside] = amr.TagVal.SET

    assert tba.has_tags(tag_box)
    assert not tba.has_tags(amr.Box(amr.IntVect(48), amr.IntVect(55)))


def test_tagboxarray_buffer_coarsen(boxarr, distmap):
    tba = amr.TagBoxArray(boxarr, distmap, 2)
    sub_box = amr.Box(amr.IntVect(8), amr.IntVect(15))
    tba.set_val(amr.BoxArray(sub_box), amr.TagVal.SET)

    # buffering grows the tagged region
    grown_box = amr.Box(amr.IntVect(7), amr.IntVect(16))
    assert not tba.has_tags(amr.Box(amr.IntVect(7), amr.IntVect(7)))
    tba.buffer(amr.IntVect(1))
    assert tba.has_tags(grown_box)

    # coarsening shrinks the index space of the tags
    tba.coarsen(amr.IntVect(2))
    assert tba.has_tags(amr.Box(amr.IntVect(4), amr.IntVect(7)))
