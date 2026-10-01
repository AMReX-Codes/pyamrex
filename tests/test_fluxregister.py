# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def make_geom(box):
    sd = amr.Config.spacedim
    real_box = amr.RealBox([0.0] * sd, [1.0] * sd)
    return amr.Geometry(box, real_box, 0, [0] * sd)


def setup():
    """A coarse level covering the domain and a fine level covering the
    center region"""
    sd = amr.Config.spacedim
    n_cell = 8
    ratio = amr.IntVect(2)

    crse_dom = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    ba_c = amr.BoxArray(crse_dom)
    dm_c = amr.DistributionMapping(ba_c)

    fine_box = amr.Box(amr.IntVect(4), amr.IntVect(11))  # cells 2..5 refined
    ba_f = amr.BoxArray(fine_box)
    dm_f = amr.DistributionMapping(ba_f)

    ncomp = 1
    fr = amr.FluxRegister(ba_f, dm_f, ratio, 1, ncomp)
    return sd, crse_dom, ba_c, dm_c, ba_f, dm_f, fr


def face_mfab(ba, dm, dir, val):
    ba_face = amr.BoxArray(ba)
    ba_face.surroundingNodes(dir)
    mf = amr.MultiFab(ba_face, dm, 1, 0)
    mf.set_val(val)
    return mf


def test_fluxregister_construct():
    sd, crse_dom, ba_c, dm_c, ba_f, dm_f, fr = setup()
    assert fr.fine_level == 1
    assert fr.crse_level == 0
    assert fr.n_comp == 1
    assert fr.ref_ratio == amr.IntVect(2)

    fr.set_val(0.0)
    assert np.isclose(fr.sum_reg(0), 0.0)


def test_fluxregister_matched_fluxes_cancel():
    """coarse and consistently scaled fine fluxes cancel: no correction"""
    sd, crse_dom, ba_c, dm_c, ba_f, dm_f, fr = setup()
    cgeom = make_geom(crse_dom)
    ratio = 2

    fr.set_val(0.0)

    c = 3.0
    for d in range(sd):
        cflux = face_mfab(ba_c, dm_c, d, c)
        # integrated coarse flux per coarse face
        fr.crse_init(cflux, d, 0, 0, 1, mult=-1.0)
        # each coarse face consists of ratio^(sd-1) fine faces
        fflux = face_mfab(ba_f, dm_f, d, c / ratio ** (sd - 1))
        fr.fine_add(fflux, d, 0, 0, 1, mult=1.0)

    assert np.isclose(fr.sum_reg(0), 0.0)

    # refluxing with a zero register leaves the solution unchanged
    phi = amr.MultiFab(ba_c, dm_c, 1, 0)
    phi.set_val(1.0)
    fr.reflux(phi, 1.0, 0, 0, 1, cgeom)
    assert np.isclose(phi.min(0), 1.0)
    assert np.isclose(phi.max(0), 1.0)


def test_fluxregister_mismatch_corrects():
    """mismatched fine fluxes produce a non-zero correction"""
    sd, crse_dom, ba_c, dm_c, ba_f, dm_f, fr = setup()
    cgeom = make_geom(crse_dom)
    ratio = 2

    fr.set_val(0.0)

    c = 3.0
    d = 0  # only x-direction fluxes
    cflux = face_mfab(ba_c, dm_c, d, c)
    fr.crse_init(cflux, d, 0, 0, 1, mult=-1.0)
    # fine fluxes twice as large as the matched value:
    # per coarse boundary face the register holds -c + 2c = c
    fflux = face_mfab(ba_f, dm_f, d, 2.0 * c / ratio ** (sd - 1))
    fr.fine_add(fflux, d, 0, 0, 1, mult=1.0)

    # note: sum_reg sums lo faces minus hi faces, which cancels for a
    # constant register; the local corrections below do not
    assert np.isclose(fr.sum_reg(0), 0.0)

    phi = amr.MultiFab(ba_c, dm_c, 1, 0)
    phi.set_val(1.0)
    sum_before = phi.sum(0)
    fr.reflux(phi, 1.0, 0, 0, 1, cgeom)
    # the correction changed the coarse cells next to the fine boundary...
    assert not np.isclose(phi.max(0), 1.0)
    assert not np.isclose(phi.min(0), 1.0)
    # ...but conserves the total: the same constant flux mismatch enters
    # on the low side and leaves on the high side of the fine patch
    assert np.isclose(phi.sum(0), sum_before)
