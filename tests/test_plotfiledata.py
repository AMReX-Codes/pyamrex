# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def write_test_plotfile(filename):
    """Write single-level plotfile (in order to read it back in)."""
    domain_box = amr.Box([0, 0, 0], [31, 31, 31])
    real_box = amr.RealBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    geom = amr.Geometry(domain_box, real_box, amr.CoordSys.cartesian, [0, 0, 0])

    ba = amr.BoxArray(domain_box)
    dm = amr.DistributionMapping(ba, 1)
    mf = amr.MultiFab(ba, dm, 1, 0)
    mf.set_val(np.pi)

    time = 1.0
    level_step = 200
    var_names = amr.Vector_string(["density"])
    amr.write_single_level_plotfile(filename, mf, var_names, geom, time, level_step)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_plotfiledata_read():
    """Generate and then read plotfile using PlotFileUtil bindings."""
    plt_filename = "test_plt00200"
    write_test_plotfile(plt_filename)
    plt = amr.PlotFileData(plt_filename)

    assert plt.spaceDim() == 3
    assert plt.time() == 1.0
    assert plt.finestLevel() == 0
    assert plt.refRatio(0) == 0
    assert plt.coordSys() == amr.CoordSys.cartesian

    probDomain = plt.probDomain(0)
    probSize = plt.probSize()
    probLo = plt.probLo()
    probHi = plt.probHi()
    cellSize = plt.cellSize(0)
    varNames = plt.varNames()
    nComp = plt.nComp()
    nGrowVect = plt.nGrowVect(0)

    assert probDomain.small_end == amr.IntVect(0, 0, 0)
    assert probDomain.big_end == amr.IntVect(31, 31, 31)

    assert probSize == [1.0, 1.0, 1.0]
    assert probLo == [-0.5, -0.5, -0.5]
    assert probHi == [0.5, 0.5, 0.5]
    assert cellSize == [1.0 / 32.0, 1.0 / 32.0, 1.0 / 32.0]
    assert varNames == amr.Vector_string(["density"])
    assert nComp == 1
    assert nGrowVect == amr.IntVect(0, 0, 0)

    for compname in varNames:
        mfab_comp = plt.get(0, compname)
        nboxes = 0

        for mfi in mfab_comp:
            marr = mfab_comp.array(mfi)
            # numpy/cupy representation: non-copying view, including the
            # guard/ghost region
            marr_xp = marr.to_xp()
            assert marr_xp.shape == (32, 32, 32, 1)
            assert np.all(marr_xp[:, :, :, :] == np.pi)
            nboxes += 1

        assert nboxes == 1


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_plotfiledata_read_docs():
    """Read a plotfile: compact example included in the manual."""
    plt_filename = "test_plt00200_docs"
    write_test_plotfile(plt_filename)

    # Manual: Read Plotfile Mesh START
    plt = amr.PlotFileData(plt_filename)

    # meta-data: AMR levels, domain extent and cell sizes
    finest_level = plt.finestLevel()
    prob_lo = plt.probLo()  # physical coordinates of the lower domain corner
    prob_hi = plt.probHi()  # ... and the upper domain corner
    cell_size = plt.cellSize(0)  # cell sizes (dx, dy, dz) on level 0
    var_names = plt.varNames()  # stored field components, e.g., ["density"]

    # read a field component on a level as a MultiFab ...
    mf_density = plt.get(0, "density")

    # ... and access its blocks as numpy/cupy/dpnp arrays (zero-copy views)
    total = 0.0
    for mfi in mf_density:
        marr_xp = mf_density.array(mfi).to_xp()
        # float() coerces the per-block reduction to a host scalar for any
        # array module (NumPy/CuPy/dpnp)
        total += float(marr_xp.sum())  # compute, plot, analyze, ...
    # Manual: Read Plotfile Mesh END

    assert finest_level == 0
    assert prob_lo == [-0.5, -0.5, -0.5]
    assert prob_hi == [0.5, 0.5, 0.5]
    assert cell_size == [1.0 / 32.0] * 3
    assert var_names == amr.Vector_string(["density"])
    assert np.isclose(total, np.pi * 32**3)
