# -*- coding: utf-8 -*-

import numpy
import pytest

import amrex.space3d as amr


@pytest.mark.skipif(not amr.Config.have_eb, reason="Requires -DAMReX_EB=ON")
def test_makeEBFabFactory():
    n_cell = 64
    max_grid_size = 16

    # Build Geometry
    domain = amr.Box(
        amr.IntVect(0, 0, 0), amr.IntVect(n_cell - 1, n_cell - 1, n_cell - 1)
    )
    real_box = amr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    coord = 0  # Cartesian
    is_per = [1, 1, 1]  # is periodic?
    geom = amr.Geometry(domain, real_box, coord, is_per)

    # EB parameters
    pp = amr.ParmParse("eb2")
    pp.add("geom_type", "sphere")
    pp.addarr("sphere_center", [0.5, 0.5, 0.5])
    rsphere = 0.25
    pp.add("sphere_radius", rsphere)
    pp.add("sphere_has_fluid_inside", 1)

    # EB generation
    eb_requried_level = 0
    eb_max_level = 2
    amr.EB2_Build(geom, eb_requried_level, eb_max_level)

    # Build BoxArray
    ba = amr.BoxArray(domain)
    ba.max_size(max_grid_size)

    # Build DistributionMapping
    dm = amr.DistributionMapping(ba)

    # Make EB Factory
    ng = amr.Vector_int([1, 1, 1])
    factory = amr.makeEBFabFactory(geom, ba, dm, ng, amr.EBSupport.full)

    # Get EB data
    vfrac = factory.getVolFrac()

    dx = geom.data().CellSize()
    total_vol = vfrac.sum() * dx[0] * dx[1] * dx[2]
    sphere_vol = 4.0 / 3.0 * numpy.pi * rsphere**3
    assert abs(sphere_vol - total_vol) / sphere_vol < 2.0e-3
