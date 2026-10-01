# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def make_geom(box):
    """Non-periodic Cartesian unit-box Geometry"""
    sd = amr.Config.spacedim
    real_box = amr.RealBox([0.0] * sd, [1.0] * sd)
    return amr.Geometry(box, real_box, 0, [0] * sd)


def fill_global_linear_x(mf):
    """f(i,j,k) = i (global cell index in x), filled incl. ghost cells"""
    for mfi in mf:
        bx = mfi.validbox()
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        ngx = mf.n_grow_vect[0]
        i = np.arange(bx.small_end[0] - ngx, bx.small_end[0] - ngx + arr.shape[0])
        arr[...] = i[(slice(None),) + (np.newaxis,) * 3]


def test_average_down_cell():
    """average_down conserves the (volume-weighted) sum"""
    sd = amr.Config.spacedim
    ratio = 2

    crse_dom = amr.Box(amr.IntVect(0), amr.IntVect(15))
    fine_dom = amr.Box(amr.IntVect(0), amr.IntVect(2 * 15 + 1))

    ba_f = amr.BoxArray(fine_dom)
    ba_f.max_size(16)
    dm_f = amr.DistributionMapping(ba_f)
    mf_f = amr.MultiFab(ba_f, dm_f, 1, 0)
    fill_global_linear_x(mf_f)

    ba_c = amr.BoxArray(crse_dom)
    ba_c.max_size(8)
    dm_c = amr.DistributionMapping(ba_c)
    mf_c = amr.MultiFab(ba_c, dm_c, 1, 0)
    mf_c.set_val(-42.0)

    # with geometries (volume weighted; arithmetic mean in Cartesian)
    amr.average_down(mf_f, mf_c, make_geom(fine_dom), make_geom(crse_dom), 0, 1, ratio)
    assert np.isclose(mf_c.sum(0) * ratio**sd, mf_f.sum(0))

    # without geometries, IntVect ratio
    mf_c.set_val(-42.0)
    amr.average_down(mf_f, mf_c, 0, 1, amr.IntVect(ratio))
    assert np.isclose(mf_c.sum(0) * ratio**sd, mf_f.sum(0))

    # without geometries, int ratio
    mf_c.set_val(-42.0)
    amr.average_down(mf_f, mf_c, 0, 1, ratio)
    assert np.isclose(mf_c.sum(0) * ratio**sd, mf_f.sum(0))


def test_average_cellcenter_to_face_roundtrip():
    """cc -> face -> cc is exact for linear data"""
    sd = amr.Config.spacedim
    box = amr.Box(amr.IntVect(0), amr.IntVect(15))
    geom = make_geom(box)
    ba = amr.BoxArray(box)
    dm = amr.DistributionMapping(ba)

    # average_cellcenter_to_face requires >= 1 ghost cell on cc
    cc = amr.MultiFab(ba, dm, 1, 1)
    fill_global_linear_x(cc)

    fc = []
    for d in range(sd):
        ba_face = amr.BoxArray(ba)
        ba_face.surroundingNodes(d)
        fc.append(amr.MultiFab(ba_face, dm, 1, 0))
        fc[d].set_val(-42.0)

    amr.average_cellcenter_to_face(fc, cc, geom)

    # interior x-face value at face index i is the midpoint between
    # cells i-1 and i: f = i - 0.5
    for mfi in fc[0]:
        bx = mfi.validbox()
        arr = fc[0].array(mfi).to_xp(copy=False, order="F")
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        expected = (i - 0.5)[(slice(None),) + (np.newaxis,) * 3]
        # skip the domain boundary faces (filled from ghost cells)
        assert np.allclose(arr[1:-1, ...], expected[1:-1, ...])

    # round trip back to cell centers: face -> cc produces a vector
    # field with one component per direction
    cc2 = amr.MultiFab(ba, dm, sd, 0)
    cc2.set_val(-42.0)
    amr.average_face_to_cellcenter(cc2, 0, fc)
    for mfi in cc2:
        bx = mfi.validbox()
        arr = cc2.array(mfi).to_xp(copy=False, order="F")
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        # the d-faces of a cell average back to the cell value i for
        # data linear in x in every direction d
        expected = i[(slice(None),) + (np.newaxis,) * 3]
        for d in range(sd):
            assert np.allclose(arr[..., d], expected[..., 0])

    # invalid arguments raise instead of corrupting memory
    if sd > 1:
        # too few components in cc
        cc_bad = amr.MultiFab(ba, dm, 1, 0)
        with pytest.raises(ValueError):
            amr.average_face_to_cellcenter(cc_bad, 0, fc)
        # too few per-direction MultiFabs
        with pytest.raises(ValueError):
            amr.average_face_to_cellcenter(cc2, 0, fc[: sd - 1])


def test_average_node_to_cellcenter():
    """nodal -> cc averaging is exact for linear data"""
    box = amr.Box(amr.IntVect(0), amr.IntVect(15))
    ba = amr.BoxArray(box)
    dm = amr.DistributionMapping(ba)

    ba_nd = amr.BoxArray(ba)
    ba_nd.surroundingNodes()
    nd = amr.MultiFab(ba_nd, dm, 1, 0)
    # f = i at the nodes (linear in x)
    for mfi in nd:
        bx = mfi.validbox()
        arr = nd.array(mfi).to_xp(copy=False, order="F")
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        arr[...] = i[(slice(None),) + (np.newaxis,) * 3]

    cc = amr.MultiFab(ba, dm, 1, 0)
    cc.set_val(-42.0)
    amr.average_node_to_cellcenter(cc, 0, nd, 0, 1)

    for mfi in cc:
        bx = mfi.validbox()
        arr = cc.array(mfi).to_xp(copy=False, order="F")
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        expected = (i + 0.5)[(slice(None),) + (np.newaxis,) * 3]
        assert np.allclose(arr, expected)

    # out-of-range components raise instead of corrupting memory
    with pytest.raises(ValueError):
        amr.average_node_to_cellcenter(cc, 1, nd, 0, 1)
    with pytest.raises(ValueError):
        amr.average_node_to_cellcenter(cc, 0, nd, 1, 1)


def test_average_down_faces():
    """average_down_faces conserves a constant field"""
    sd = amr.Config.spacedim

    crse_dom = amr.Box(amr.IntVect(0), amr.IntVect(7))
    fine_dom = amr.Box(amr.IntVect(0), amr.IntVect(15))

    fine, crse = [], []
    for d in range(sd):
        ba_f = amr.BoxArray(fine_dom)
        ba_f.surroundingNodes(d)
        dm_f = amr.DistributionMapping(ba_f)
        fine.append(amr.MultiFab(ba_f, dm_f, 1, 0))
        fine[d].set_val(7.0)

        ba_c = amr.BoxArray(crse_dom)
        ba_c.surroundingNodes(d)
        dm_c = amr.DistributionMapping(ba_c)
        crse.append(amr.MultiFab(ba_c, dm_c, 1, 0))
        crse[d].set_val(-42.0)

    amr.average_down_faces(fine, crse, amr.IntVect(2))
    for d in range(sd):
        assert np.isclose(crse[d].min(0), 7.0)
        assert np.isclose(crse[d].max(0), 7.0)


def test_write_multi_level_plotfile(tmpdir):
    """write a two-level plotfile and read it back"""
    plt = str(tmpdir.join("plt_ml00000"))
    if amr.Config.have_mpi:
        # pytest creates a different tmpdir on every rank: broadcast the
        # path so all ranks write to (and read from) the same plotfile
        from mpi4py import MPI

        plt = MPI.COMM_WORLD.bcast(plt, root=0)

    domains = [
        amr.Box(amr.IntVect(0), amr.IntVect(15)),
        amr.Box(amr.IntVect(0), amr.IntVect(31)),
    ]
    mfs, geoms = [], []
    for lev, dom in enumerate(domains):
        ba = amr.BoxArray(dom)
        ba.max_size(16)
        dm = amr.DistributionMapping(ba)
        mf = amr.MultiFab(ba, dm, 1, 0)
        mf.set_val(1.0 + lev)
        mfs.append(mf)
        geoms.append(make_geom(dom))

    amr.write_multi_level_plotfile(
        plt,
        mfs,
        ["phi"],
        geoms,
        0.5,
        [10, 10],
        [amr.IntVect(2)],
    )

    pfd = amr.PlotFileData(plt)
    assert pfd.finestLevel() == 1
    assert list(pfd.varNames()) == ["phi"]
    assert np.isclose(pfd.time(), 0.5)
    assert pfd.levelStep(0) == 10
    for lev in range(2):
        mf_read = pfd.get(lev, "phi")
        assert np.isclose(mf_read.min(0), 1.0 + lev)
        assert np.isclose(mf_read.max(0), 1.0 + lev)


def test_io_path_helpers():
    assert amr.level_path(5) == "Level_5"
    assert amr.multifab_header_path(5) == "Level_5/Cell"
    assert amr.level_full_path(5, "plt00005") == "plt00005/Level_5"
    assert amr.multifab_file_full_prefix(5, "plt00005") == "plt00005/Level_5/Cell"


def test_directory_helpers(tmpdir):
    path = str(tmpdir.join("a", "b"))
    assert not amr.file_exists(path)
    assert amr.util_create_directory(path)
    assert amr.file_exists(path)
    # rename the old one, create a new one
    amr.util_create_clean_directory(path)
    assert amr.file_exists(path)
    # remove the old one, create a new one
    amr.util_create_directory_destructive(path)
    assert amr.file_exists(path)
