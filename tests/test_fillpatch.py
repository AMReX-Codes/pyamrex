# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def make_geom(box, periodic=True):
    """Cartesian unit-box Geometry"""
    sd = amr.Config.spacedim
    real_box = amr.RealBox([0.0] * sd, [1.0] * sd)
    return amr.Geometry(box, real_box, 0, [1 if periodic else 0] * sd)


def fill_global_linear_x(mf):
    """f(i,j,k) = i (global cell index in x), filled incl. ghost cells"""
    for mfi in mf:
        bx = mfi.validbox()
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        ngx = mf.n_grow_vect[0]
        i = np.arange(bx.small_end[0] - ngx, bx.small_end[0] - ngx + arr.shape[0])
        arr[...] = i[(slice(None),) + (np.newaxis,) * 3]


def int_dir_bcs():
    sd = amr.Config.spacedim
    return amr.Vector_BCRec(
        [amr.BCRec(lo=[amr.BCType.int_dir] * sd, hi=[amr.BCType.int_dir] * sd)]
    )


def test_fill_patch_single_level():
    """fill valid + periodic ghost cells from a source MultiFab"""
    n_cell = 16
    box = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    ba.max_size(8)
    dm = amr.DistributionMapping(ba)

    src = amr.MultiFab(ba, dm, 1, 0)
    fill_global_linear_x(src)

    dst = amr.MultiFab(ba, dm, 1, 1)
    dst.set_val(-42.0)

    physbc = amr.PhysBCFunctNoOp()
    amr.fill_patch_single_level(dst, 0.0, [src], [0.0], 0, 0, 1, geom, physbc, 0)

    for mfi in dst:
        bx = mfi.validbox()
        arr = dst.array(mfi).to_xp(copy=False, order="F")
        # global x indices of the view, including the ghost cells
        i = np.arange(bx.small_end[0] - 1, bx.big_end[0] + 2)
        # periodic wrap-around
        expected = i % n_cell
        assert np.allclose(arr, expected[(slice(None),) + (np.newaxis,) * 3])


def test_fill_patch_single_level_time_interp():
    """source data at two times is interpolated linearly in time"""
    box = amr.Box(amr.IntVect(0), amr.IntVect(7))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    dm = amr.DistributionMapping(ba)

    src_old = amr.MultiFab(ba, dm, 1, 0)
    src_old.set_val(1.0)
    src_new = amr.MultiFab(ba, dm, 1, 0)
    src_new.set_val(3.0)

    dst = amr.MultiFab(ba, dm, 1, 1)
    physbc = amr.PhysBCFunctNoOp()
    amr.fill_patch_single_level(
        dst, 0.5, [src_old, src_new], [0.0, 1.0], 0, 0, 1, geom, physbc, 0
    )
    assert np.isclose(dst.min(0), 2.0)
    assert np.isclose(dst.max(0), 2.0)


def test_interp_from_coarse_level():
    """interpolation from a coarse level: exact for constant data;
    exact in the interior for linear data"""
    n_cell = 16

    crse_dom = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    fine_dom = amr.Box(amr.IntVect(0), amr.IntVect(2 * n_cell - 1))
    cgeom = make_geom(crse_dom)
    fgeom = make_geom(fine_dom)

    ba_c = amr.BoxArray(crse_dom)
    dm_c = amr.DistributionMapping(ba_c)
    crse = amr.MultiFab(ba_c, dm_c, 1, 1)
    fill_global_linear_x(crse)

    # fine patch covering the center of the domain
    fine_box = amr.Box(amr.IntVect(n_cell // 2), amr.IntVect(3 * n_cell // 2 - 1))
    ba_f = amr.BoxArray(fine_box)
    dm_f = amr.DistributionMapping(ba_f)
    fine = amr.MultiFab(ba_f, dm_f, 1, 0)
    fine.set_val(-42.0)

    physbc = amr.PhysBCFunctNoOp()
    amr.interp_from_coarse_level(
        fine,
        0.0,
        crse,
        0,
        0,
        1,
        cgeom,
        fgeom,
        physbc,
        0,
        physbc,
        0,
        amr.IntVect(2),
        amr.cell_cons_interp,
        int_dir_bcs(),
        0,
    )

    # for f(i_c) = i_c, linear conservative interpolation gives
    # f(i_f) = i_f / 2 - 1/4 away from limited slopes
    for mfi in fine:
        bx = mfi.validbox()
        arr = fine.array(mfi).to_xp(copy=False, order="F")
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        expected = (i / 2.0 - 0.25)[(slice(None),) + (np.newaxis,) * 3]
        assert np.allclose(arr, expected)

    # the same, dispatched through the MFInterpolater overload set
    fine2 = amr.MultiFab(ba_f, dm_f, 1, 0)
    fine2.set_val(-42.0)
    amr.interp_from_coarse_level(
        fine2,
        0.0,
        crse,
        0,
        0,
        1,
        cgeom,
        fgeom,
        physbc,
        0,
        physbc,
        0,
        amr.IntVect(2),
        amr.mf_cell_cons_interp,
        int_dir_bcs(),
        0,
    )
    # mean conservation: both interpolations preserve the coarse mean
    assert np.isclose(fine2.sum(0), fine.sum(0))

    # conservation: averaging the interpolated fine data back down
    # reproduces the coarse data on the covered region
    crse_check = crse.copy()
    amr.average_down(fine, crse_check, 0, 1, amr.IntVect(2))
    # same BoxArray/DistributionMapping: one MFIter indexes both
    for mfi in crse:
        a = crse.array(mfi).to_xp(copy=False, order="F")
        b = crse_check.array(mfi).to_xp(copy=False, order="F")
        assert np.allclose(a, b)


def test_fill_patch_two_levels():
    """valid cells come from the fine data, ghost cells outside the fine
    region from coarse interpolation"""
    n_cell = 16

    crse_dom = amr.Box(amr.IntVect(0), amr.IntVect(n_cell - 1))
    fine_dom = amr.Box(amr.IntVect(0), amr.IntVect(2 * n_cell - 1))
    cgeom = make_geom(crse_dom)
    fgeom = make_geom(fine_dom)

    ba_c = amr.BoxArray(crse_dom)
    dm_c = amr.DistributionMapping(ba_c)
    crse = amr.MultiFab(ba_c, dm_c, 1, 1)
    crse.set_val(7.0)

    fine_box = amr.Box(amr.IntVect(n_cell // 2), amr.IntVect(3 * n_cell // 2 - 1))
    ba_f = amr.BoxArray(fine_box)
    dm_f = amr.DistributionMapping(ba_f)
    fine_src = amr.MultiFab(ba_f, dm_f, 1, 0)
    fine_src.set_val(7.0)

    dst = amr.MultiFab(ba_f, dm_f, 1, 2)
    dst.set_val(-42.0)

    physbc = amr.PhysBCFunctNoOp()
    amr.fill_patch_two_levels(
        dst,
        0.0,
        [crse],
        [0.0],
        [fine_src],
        [0.0],
        0,
        0,
        1,
        cgeom,
        fgeom,
        physbc,
        0,
        physbc,
        0,
        amr.IntVect(2),
        amr.cell_cons_interp,
        int_dir_bcs(),
        0,
    )

    # for a globally constant field, valid and ghost cells (filled from
    # the coarse level) are all the constant
    for mfi in dst:
        arr = dst.array(mfi).to_xp(copy=False, order="F")
        assert np.allclose(arr, 7.0)


def test_fill_patch_user_bc_called():
    """the Python physical boundary functor is invoked by fill_patch"""
    box = amr.Box(amr.IntVect(0), amr.IntVect(7))
    geom = make_geom(box, periodic=False)
    ba = amr.BoxArray(box)
    dm = amr.DistributionMapping(ba)

    src = amr.MultiFab(ba, dm, 1, 0)
    src.set_val(1.0)
    dst = amr.MultiFab(ba, dm, 1, 1)
    dst.set_val(-42.0)

    called = []

    def fill(mf, dcomp, ncomp, nghost, time, bccomp):
        called.append((dcomp, ncomp, time, bccomp))
        # fill all (domain boundary) ghost cells with a marker value
        for mfi in mf:
            arr = mf.array(mfi).to_xp(copy=False, order="F")
            arr[arr == -42.0] = 99.0

    physbc = amr.PhysBCFunctUser(fill)
    amr.fill_patch_single_level(dst, 0.25, [src], [0.0], 0, 0, 1, geom, physbc, 0)

    assert len(called) == 1
    assert called[0][0] == 0  # dcomp
    assert called[0][1] == 1  # ncomp
    assert np.isclose(called[0][2], 0.25)  # time
    # valid cells from src, ghost cells from the Python callback
    assert np.isclose(dst.min(0, nghost=1), 1.0)
    assert np.isclose(dst.max(0, nghost=1), 99.0)
