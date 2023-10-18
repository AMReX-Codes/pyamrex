# -*- coding: utf-8 -*-

import amrex.space3d as amr

# import amrex.space3d as amr.CoordSys.CoordType as CoordType


def test_coordSys_coordType():
    CType = amr.CoordSys.CoordType
    cs = amr.CoordSys()

    print(cs.ok())
    assert not cs.ok()
    print(cs.Coord())
    assert cs.Coord() == CType.undef
    print(cs.CoordInt())
    assert cs.CoordInt() == -1

    cs.SetCoord(CType.cartesian)
    assert cs.Coord() == CType.cartesian
    assert cs.CoordInt() == 0

    cs.SetCoord(CType.RZ)
    assert cs.Coord() == CType.RZ
    assert cs.CoordInt() == 1

    cs.SetCoord(CType.SPHERICAL)
    assert cs.Coord() == CType.SPHERICAL
    assert cs.CoordInt() == 2
