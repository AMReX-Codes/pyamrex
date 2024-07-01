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
