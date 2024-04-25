# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr
from amrex.space3d import Geometry as Gm

# TODO: return coord?


@pytest.fixture(scope="function")
def box():
    return amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(127, 127, 127))


@pytest.fixture(scope="function")
def real_box():
    return amr.RealBox([0, 0, 0], [1, 2, 5])


@pytest.fixture(scope="function")
def geometry(box, real_box):
    coord = 1
    is_periodic = [0, 0, 1]
    return amr.Geometry(box, real_box, coord, is_periodic)


def test_geometry_data(box, real_box):
    gd = amr.GeometryData()
    print(gd.coord)
    assert gd.coord == 0
    print(gd.domain)
    print(gd.domain.small_end)
    assert gd.domain.small_end == amr.IntVect(1, 1, 1)
    print(gd.domain.big_end)
    assert gd.domain.big_end == amr.IntVect(0, 0, 0)
    print(gd.prob_domain)
    assert amr.AlmostEqual(gd.prob_domain, amr.RealBox(0, 0, 0, -1, -1, -1))

    print(gd.isPeriodic())
    print(gd.is_periodic)
    assert gd.is_periodic == gd.isPeriodic() == [0, 0, 0]

    with pytest.raises(AttributeError):
        gd.coord = 1
    with pytest.raises(AttributeError):
        gd.domain = box
    with pytest.raises(AttributeError):
        gd.prob_domain = real_box
    with pytest.raises(AttributeError):
        gd.is_periodic = [1, 0, 1]
    with pytest.raises(AttributeError):
        gd.dx = [0.1, 0.2, 0.3]


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_geometry_define(box, real_box):
    gm = Gm()
    coord = 1
    is_periodic = [0, 0, 1]
    gm.define(box, real_box, coord, is_periodic)
    assert gm.ProbLength(0) == 1 and gm.ProbLength(1) == 2 and gm.ProbLength(2) == 5
    assert gm.isPeriodic() == is_periodic


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_probDomain(box, real_box):
    gm = Gm()
    coord = 1
    is_periodic = [0, 0, 1]
    gm.define(box, real_box, coord, is_periodic)
    assert gm.ok()

    lo = [0, -1, 1]
    hi = [1, 0, 2]
    rb = amr.RealBox(lo, hi)
    gm.prob_domain = rb
    assert (
        np.isclose(gm.ProbLo(0), lo[0])
        and np.isclose(gm.ProbLo(1), lo[1])
        and np.isclose(gm.ProbLo(2), lo[2])
        and np.isclose(gm.ProbHi(0), hi[0])
        and np.isclose(gm.ProbHi(1), hi[1])
        and np.isclose(gm.ProbHi(2), hi[2])
    )


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_size(geometry):
    gm = geometry

    assert np.isclose(gm.ProbSize(), 10)
    assert np.isclose(gm.ProbLength(0), 1)
    assert np.isclose(gm.ProbLength(1), 2)
    assert np.isclose(gm.ProbLength(2), 5)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_domain(box, real_box):
    bx = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(127, 127, 127))
    gm = Gm()
    coord = 1
    is_periodic = [0, 0, 1]
    gm.define(box, real_box, coord, is_periodic)
    assert gm.ok()

    gm.domain = bx
    assert gm.domain.small_end == bx.small_end and gm.domain.big_end == bx.big_end


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_periodic_queries(box, real_box):
    coord = 1
    is_periodic = [0, 0, 1]
    gm = Gm(box, real_box, coord, is_periodic)

    pdcity = gm.periodicity()  # noqa

    assert gm.isAnyPeriodic()
    assert not gm.isPeriodic(0) and not gm.isPeriodic(1) and gm.isPeriodic(2)
    assert not gm.isAllPeriodic()
    assert gm.isPeriodic() == is_periodic
    gm.setPeriodicity([0, 0, 0])
    assert not gm.isAnyPeriodic()
    gm.setPeriodicity([1, 1, 1])
    assert gm.isAllPeriodic()


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_periodicity(geometry):
    gm = geometry
    bx = gm.domain

    for _non_periodic_coord in [0, 1]:
        error_thrown = False
        try:
            gm.period(0)
        except RuntimeError:
            error_thrown = True
        assert error_thrown

    assert gm.period(2) == bx.length(2)
    pdcity = amr.Periodicity(bx.length() * amr.IntVect(gm.isPeriodic()))
    assert gm.periodicity() == pdcity

    iv1 = amr.IntVect(0, 0, 0)
    iv2 = amr.IntVect(9, 19, 29)
    pdcity = amr.Periodicity(amr.IntVect(0, 0, 30))
    bx = amr.Box(iv1, iv2)
    assert gm.periodicity(bx) == pdcity


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_grow(geometry):
    gm = geometry
    gm_grow_pd = gm.growPeriodicDomain(2)
    assert gm_grow_pd.small_end == amr.IntVect(0, 0, -2)
    assert gm_grow_pd.big_end == amr.IntVect(127, 127, 129)
    gm_grow_npd = gm.growNonPeriodicDomain(3)
    assert gm_grow_npd.small_end == amr.IntVect(-3, -3, 0)
    assert gm_grow_npd.big_end == amr.IntVect(130, 130, 127)


def test_data(geometry, box, real_box):
    geometry_data = geometry.data()
    gd = geometry_data

    assert gd.domain.small_end == box.small_end == gd.domain.small_end
    assert gd.domain.big_end == box.big_end == gd.domain.big_end

    assert amr.AlmostEqual(gd.prob_domain, real_box)
    assert np.allclose(real_box.lo(), gd.prob_domain.lo())
    assert np.allclose(real_box.hi(), gd.prob_domain.hi())
    assert (
        np.isclose(real_box.lo(0), gd.prob_domain.lo(0))
        and np.isclose(real_box.lo(1), gd.prob_domain.lo(1))
        and np.isclose(real_box.lo(2), gd.prob_domain.lo(2))
    )
    assert (
        np.isclose(real_box.hi(0), gd.prob_domain.hi(0))
        and np.isclose(real_box.hi(1), gd.prob_domain.hi(1))
        and np.isclose(real_box.hi(2), gd.prob_domain.hi(2))
    )

    assert np.allclose(real_box.lo(), gd.ProbLo())
    assert np.allclose(real_box.hi(), gd.ProbHi())
    assert (
        np.isclose(real_box.lo(0), gd.ProbLo(0))
        and np.isclose(real_box.lo(1), gd.ProbLo(1))
        and np.isclose(real_box.lo(2), gd.ProbLo(2))
    )
    assert (
        np.isclose(real_box.hi(0), gd.ProbHi(0))
        and np.isclose(real_box.hi(1), gd.ProbHi(1))
        and np.isclose(real_box.hi(2), gd.ProbHi(2))
    )

    assert gd.coord == gd.Coord() == 1

    assert gd.is_periodic == [0, 0, 1]
    assert gd.is_periodic[0] == gd.isPeriodic(0) == 0
    assert gd.is_periodic[1] == gd.isPeriodic(1) == 0
    assert gd.is_periodic[2] == gd.isPeriodic(2) == 1

    print(gd.CellSize())
    assert np.allclose(gd.CellSize(), [0.0078125, 0.015625, 0.0390625])
    assert np.isclose(gd.CellSize(0), 0.0078125)
    assert np.isclose(gd.CellSize(1), 0.015625)
    assert np.isclose(gd.CellSize(2), 0.0390625)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_coarsen_refine(geometry):
    iv1 = amr.IntVect(-1, -2, -3)
    iv2 = amr.IntVect(4, 5, 6)
    bx = amr.Box(iv1, iv2)
    rb = amr.RealBox(-1, -2, -3, 2, 4, 6)
    gmc = amr.Geometry(bx, rb, 1, [0, 0, 1])
    cv = amr.IntVect(2, 2, 1)
    gmc.coarsen(cv)
    assert gmc.domain.small_end == amr.IntVect(-1, -1, -3)
    assert gmc.domain.big_end == amr.IntVect(2, 2, 6)

    gmr = amr.Geometry(bx, rb, 1, [0, 0, 1])
    rv = amr.IntVect(2, 2, 3)
    gmr.refine(rv)
    assert gmr.domain.small_end == amr.IntVect(-2, -4, -9)
    assert gmr.domain.big_end == amr.IntVect(9, 11, 20)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_roundoff_domain():
    iv1 = amr.IntVect(-1, -2, -3)
    iv2 = amr.IntVect(4, 5, 6)
    bx = amr.Box(iv1, iv2)
    rb = amr.RealBox(0, 1.0002, 0.00105, 2, 4.2, 5)

    gm = Gm(bx, rb, 1, [0, 0, 1])

    assert gm.outsideRoundOffDomain(-1.0, 1, 2)
    assert not gm.outsideRoundOffDomain(1.00, 2.0, 2)
    assert not gm.insideRoundOffDomain(-1.0, 1, 2)
    assert gm.insideRoundOffDomain(1.00, 2.0, 2)


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_Resets():
    rb = amr.RealBox(0, 0, 0, 1, 2, 6)
    amr.Geometry.ResetDefaultProbDomain(rb)
    Gm.ResetDefaultProbDomain(rb)
    is_periodic = [1, 0, 1]
    Gm.ResetDefaultPeriodicity(is_periodic)
    Gm.ResetDefaultCoord(1)

    gm = Gm()
    assert (
        np.isclose(gm.ProbLength(0), 1)
        and np.isclose(gm.ProbLength(1), 2)
        and np.isclose(gm.ProbLength(2), 6)
    )
    assert gm.isPeriodic() == is_periodic
    print(gm.Coord())
    CType = amr.CoordSys.CoordType
    assert gm.Coord() == CType.RZ
