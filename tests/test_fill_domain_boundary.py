# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def make_geometry(box):
    """Non-periodic Cartesian unit-box Geometry"""
    sd = amr.Config.spacedim
    real_box = amr.RealBox([0.0] * sd, [1.0] * sd)
    coord = 0  # Cartesian
    is_per = [0] * sd
    return amr.Geometry(box, real_box, coord, is_per)


def valid_slices(arr_shape, n_grow_vect):
    """Slices into the valid (non-ghost) region of a 4-axis (x,y,z,n)
    Array4 view, for any AMREX_SPACEDIM"""
    sd = amr.Config.spacedim
    ng = [n_grow_vect[d] if d < sd else 0 for d in range(3)] + [0]
    return tuple(slice(g, s - g) for g, s in zip(ng, arr_shape))


def test_fill_domain_boundary_foextrap(std_box):
    """foextrap of a constant field fills all ghost cells with the constant"""
    sd = amr.Config.spacedim
    geom = make_geometry(std_box)
    ba = amr.BoxArray(std_box)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 1)

    sentinel = -42.0
    inner = 7.0

    # fill everything (incl. ghost cells) with the sentinel, then the
    # valid cells with a constant
    mf.set_val(sentinel)
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        arr[valid_slices(arr.shape, mf.n_grow_vect)] = inner

    # first fill interior box-box ghost cells: domain corner ghosts
    # extrapolate from them (same order as the C++ tutorials)
    mf.fill_boundary()

    bc = amr.Vector_BCRec(
        [amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.foextrap] * sd)]
    )
    amr.fill_domain_boundary(mf, geom, bc)

    # first-order extrapolation of a constant is the constant, everywhere
    # (domain face, edge and corner ghost cells alike)
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        assert np.all(arr == inner)


def test_fill_domain_boundary_linear(std_box):
    """foextrap continues a linear-in-x field flat at the x faces"""
    sd = amr.Config.spacedim
    geom = make_geometry(std_box)
    ba = amr.BoxArray(std_box)  # single box
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 1)
    mf.set_val(0.0)

    for mfi in mf:
        bx = mfi.validbox()
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        vs = valid_slices(arr.shape, mf.n_grow_vect)
        # f(i,j,k) = i (cell index in x)
        i = np.arange(bx.small_end[0], bx.big_end[0] + 1)
        arr[vs] = i[(slice(None),) + (np.newaxis,) * 3]

    bc = amr.Vector_BCRec(
        [amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.foextrap] * sd)]
    )
    amr.fill_domain_boundary(mf, geom, bc)

    for mfi in mf:
        bx = mfi.validbox()
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        vs = valid_slices(arr.shape, mf.n_grow_vect)
        # x-lo face ghosts equal the first valid cell (flat extrapolation)
        lo_ghost = arr[(slice(0, 1),) + vs[1:]]
        assert np.all(lo_ghost == bx.small_end[0])
        # x-hi face ghosts equal the last valid cell
        hi_ghost = arr[(slice(-1, None),) + vs[1:]]
        assert np.all(hi_ghost == bx.big_end[0])


def test_physbcfunct_noop(std_box):
    """PhysBCFunctNoOp leaves ghost cells untouched"""
    ba = amr.BoxArray(std_box)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 1)

    sentinel = -42.0
    mf.set_val(sentinel)

    physbc = amr.PhysBCFunctNoOp()
    physbc(mf, 0, 1, mf.n_grow_vect, 0.0, 0)

    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        assert np.all(arr == sentinel)


def test_physbcfunct_cpu(std_box):
    """PhysBCFunct_CpuBndryFuncFab fills domain ghost cells like
    fill_domain_boundary"""
    sd = amr.Config.spacedim
    geom = make_geometry(std_box)
    ba = amr.BoxArray(std_box)  # single box: all ghosts are domain ghosts
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 1, 1)

    sentinel = -42.0
    inner = 7.0
    mf.set_val(sentinel)
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        arr[valid_slices(arr.shape, mf.n_grow_vect)] = inner

    bc = amr.Vector_BCRec(
        [amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.foextrap] * sd)]
    )
    bndry_func = amr.CpuBndryFuncFab()
    physbc = amr.PhysBCFunct_CpuBndryFuncFab(geom, bc, bndry_func)
    physbc(mf, 0, 1, mf.n_grow_vect, 0.0, 0)

    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        assert np.all(arr == inner)


def test_physbcfunct_cpu_with_offsets(std_box):
    sd = amr.Config.spacedim
    geom = make_geometry(std_box)
    ba = amr.BoxArray(std_box)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 2, 1)

    sentinel = -42.0
    inner = 7.0
    mf.set_val(sentinel)
    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        vs = valid_slices(arr.shape, mf.n_grow_vect)
        arr[vs[:-1] + (slice(1, 2),)] = inner

    bc = amr.Vector_BCRec(
        [
            amr.BCRec(lo=[amr.BCType.ext_dir] * sd, hi=[amr.BCType.ext_dir] * sd),
            amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.foextrap] * sd),
        ]
    )
    bndry_func = amr.CpuBndryFuncFab()
    physbc = amr.PhysBCFunct_CpuBndryFuncFab(geom, bc, bndry_func)
    physbc(mf, 1, 1, mf.n_grow_vect, 0.0, 1)

    for mfi in mf:
        arr = mf.array(mfi).to_xp(copy=False, order="F")
        assert np.all(arr[..., 0] == sentinel)
        assert np.all(arr[..., 1] == inner)


def test_physbcfunct_user(std_box):
    """PhysBCFunctUser calls back into Python with reference semantics"""
    ba = amr.BoxArray(std_box)
    dm = amr.DistributionMapping(ba)
    mf = amr.MultiFab(ba, dm, 2, 1)
    mf.set_val(0.0)

    called = {}

    def fill(mf_arg, dcomp, ncomp, nghost, time, bccomp):
        called["args"] = (dcomp, ncomp, nghost, time, bccomp)
        # mutate through the callback argument: reference semantics
        mf_arg.set_val(99.0, dcomp, ncomp)

    physbc = amr.PhysBCFunctUser(fill)
    physbc(mf, 1, 1, mf.n_grow_vect, 1.5, 2)

    assert called["args"][0] == 1
    assert called["args"][1] == 1
    assert called["args"][3] == 1.5
    assert called["args"][4] == 2
    assert np.isclose(mf.max(0), 0.0)
    assert np.isclose(mf.min(0), 0.0)
    assert np.isclose(mf.max(1), 99.0)
    assert np.isclose(mf.min(1), 99.0)


def test_physbcfunct_user_exception_propagates(std_box):
    ba = amr.BoxArray(std_box)
    mf = amr.MultiFab(ba, amr.DistributionMapping(ba), 1, 0)

    def fail(*_args):
        raise ValueError("boundary callback failed")

    physbc = amr.PhysBCFunctUser(fail)
    with pytest.raises(ValueError, match="boundary callback failed"):
        physbc(mf, 0, 1, amr.IntVect(0), 0.0, 0)


def test_physbcfunct_user_keeps_callback_alive():
    import gc
    import weakref

    class Fill:
        def __call__(self, *_args):
            pass

    callback = Fill()
    callback_ref = weakref.ref(callback)
    physbc = amr.PhysBCFunctUser(callback)
    del callback
    gc.collect()
    assert callback_ref() is not None

    del physbc
    gc.collect()
    assert callback_ref() is None
