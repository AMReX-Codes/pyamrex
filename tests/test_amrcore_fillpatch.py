# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def valid_slices(arr_shape, n_grow_vect):
    """Slices into the valid (non-ghost) region of a 4-axis (x,y,z,n)
    Array4 view"""
    sd = amr.Config.spacedim
    ng = [n_grow_vect[d] if d < sd else 0 for d in range(3)] + [0]
    return tuple(slice(g, s - g) for g, s in zip(ng, arr_shape))


class GaussianCore(amr.AmrCore):
    """A two-level AmrCore mirroring the structure of the
    Advection_AmrCore tutorial: a Gaussian profile, tagged by threshold,
    with FillPatch/InterpFromCoarseLevel-based level construction."""

    def __init__(self, n_cell=32):
        sd = amr.Config.spacedim
        rb = amr.RealBox([0.0] * sd, [1.0] * sd)
        max_level = 1
        super().__init__(
            rb,
            max_level,
            amr.Vector_int([n_cell] * sd),
            0,  # Cartesian
            amr.Vector_IntVect([amr.IntVect(2)] * max_level),
            [1] * sd,  # fully periodic
        )
        self.phi = [None] * (max_level + 1)
        self.bcs = amr.Vector_BCRec(
            [
                amr.BCRec(
                    lo=[amr.BCType.int_dir] * sd,
                    hi=[amr.BCType.int_dir] * sd,
                )
            ]
        )
        self.physbc = amr.PhysBCFunctNoOp()  # fully periodic
        self.tag_threshold = 1.0e30  # nothing tagged until lowered
        self.calls = []

    def fill_gaussian(self, lev, mf):
        """phi = 1 + exp(-(r - r_center)^2 / 0.01) at cell centers"""
        sd = amr.Config.spacedim
        geom_data = self.geom(lev).data()
        dx = geom_data.CellSize()
        for mfi in mf:
            bx = mfi.validbox()
            arr = mf.array(mfi).to_xp(copy=False, order="F")
            vs = valid_slices(arr.shape, mf.n_grow_vect)
            rsq = 0.0
            for d in range(sd):
                x = (np.arange(bx.small_end[d], bx.big_end[d] + 1) + 0.5) * dx[d]
                rsq = rsq + ((x - 0.5) ** 2).reshape(
                    [-1 if i == d else 1 for i in range(4)]
                )
            arr[vs] = 1.0 + np.exp(-rsq / 0.01)
        mf.fill_boundary(self.geom(lev).periodicity())

    # pure virtual overrides ------------------------------------------
    def make_new_level_from_scratch(self, lev, time, ba, dm):
        self.calls.append(("scratch", lev))
        self.phi[lev] = amr.MultiFab(ba, dm, 1, 1)
        self.fill_gaussian(lev, self.phi[lev])

    def make_new_level_from_coarse(self, lev, time, ba, dm):
        self.calls.append(("coarse", lev))
        mf = amr.MultiFab(ba, dm, 1, 1)
        amr.interp_from_coarse_level(
            mf,
            time,
            self.phi[lev - 1],
            0,
            0,
            1,
            self.geom(lev - 1),
            self.geom(lev),
            self.physbc,
            0,
            self.physbc,
            0,
            self.ref_ratio(lev - 1),
            amr.cell_cons_interp,
            self.bcs,
            0,
        )
        self.phi[lev] = mf

    def remake_level(self, lev, time, ba, dm):
        self.calls.append(("remake", lev))
        mf = amr.MultiFab(ba, dm, 1, 1)
        amr.fill_patch_two_levels(
            mf,
            time,
            [self.phi[lev - 1]],
            [time],
            [self.phi[lev]],
            [time],
            0,
            0,
            1,
            self.geom(lev - 1),
            self.geom(lev),
            self.physbc,
            0,
            self.physbc,
            0,
            self.ref_ratio(lev - 1),
            amr.cell_cons_interp,
            self.bcs,
            0,
        )
        self.phi[lev] = mf

    def clear_level(self, lev):
        self.calls.append(("clear", lev))
        self.phi[lev] = None

    def error_est(self, lev, tags, time, ngrow):
        self.calls.append(("error_est", lev))
        phi = self.phi[lev]
        for mfi in phi:
            phi_arr = phi.array(mfi).to_xp(copy=False, order="F")
            phi_valid = phi_arr[valid_slices(phi_arr.shape, phi.n_grow_vect)]
            tag_arr = tags.array(mfi).to_xp(copy=False, order="F")
            tag_valid = tag_arr[valid_slices(tag_arr.shape, tags.n_grow_vect)]
            tag_valid[phi_valid > self.tag_threshold] = amr.TagVal.SET


def test_amrcore_two_level_lifecycle():
    core = GaussianCore()
    sd = amr.Config.spacedim

    # nothing tagged: a single level
    core.init_from_scratch(0.0)
    assert core.finest_level == 0
    assert core.count_cells(0) == 32**sd
    assert ("scratch", 0) in core.calls

    # lower the threshold: regrid creates level 1 from the coarse level
    core.tag_threshold = 1.5
    core.regrid(0, 0.0)
    assert core.finest_level == 1
    assert ("error_est", 0) in core.calls
    assert ("coarse", 1) in core.calls
    assert core.level_defined(1)
    assert core.max_ref_ratio(0) == 2
    assert core.box_array(1).numPts > 0
    assert core.phi[1] is not None

    # conservative interpolation: averaging level 1 down reproduces the
    # coarse data exactly on the covered region
    crse_check = core.phi[0].copy()
    amr.average_down(core.phi[1], crse_check, 0, 1, core.ref_ratio(0))
    # same BoxArray/DistributionMapping: one MFIter indexes both
    for mfi in core.phi[0]:
        a = core.phi[0].array(mfi).to_xp(copy=False, order="F")
        b = crse_check.array(mfi).to_xp(copy=False, order="F")
        assert np.allclose(a, b)

    # interpolated maxima cannot overshoot the coarse maxima (limiter)
    assert core.phi[1].max(0) <= core.phi[0].max(0) + 1.0e-12

    # lower the threshold further: the tagged region grows and level 1
    # is remade on the new, larger BoxArray
    # (a regrid with unchanged grids does not call remake_level)
    core.tag_threshold = 1.2
    core.regrid(0, 0.0)
    assert ("remake", 1) in core.calls
    assert core.finest_level == 1

    # raise the threshold: regrid removes level 1
    core.tag_threshold = 1.0e30
    core.regrid(0, 0.0)
    assert core.finest_level == 0
    assert ("clear", 1) in core.calls
    assert core.phi[1] is None
