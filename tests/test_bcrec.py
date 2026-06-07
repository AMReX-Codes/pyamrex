# -*- coding: utf-8 -*-

import amrex.space3d as amr


def test_bctype_values():
    # values are fixed in AMReX_BC_TYPES.H
    assert int(amr.BCType.bogus) == -666
    assert int(amr.BCType.reflect_odd) == -1
    assert int(amr.BCType.int_dir) == 0
    assert int(amr.BCType.reflect_even) == 1
    assert int(amr.BCType.foextrap) == 2
    assert int(amr.BCType.ext_dir) == 3
    assert int(amr.BCType.hoextrap) == 4
    assert int(amr.BCType.hoextrapcc) == 5
    assert int(amr.BCType.ext_dir_cc) == 6
    assert int(amr.BCType.direction_dependent) == 7


def test_physbctype_values():
    assert int(amr.PhysBCType.interior) == 0
    assert int(amr.PhysBCType.inflow) == 1
    assert int(amr.PhysBCType.outflow) == 2
    assert int(amr.PhysBCType.symmetry) == 3
    assert int(amr.PhysBCType.slipwall) == 4
    assert int(amr.PhysBCType.noslipwall) == 5
    assert int(amr.PhysBCType.inflowoutflow) == 6


def test_bcrec_default():
    bcr = amr.BCRec()
    sd = amr.Config.spacedim
    assert bcr.lo() == [amr.BCType.bogus] * sd
    assert bcr.hi() == [amr.BCType.bogus] * sd


def test_bcrec_lo_hi():
    sd = amr.Config.spacedim
    bcr = amr.BCRec(
        lo=[amr.BCType.int_dir] * sd,
        hi=[amr.BCType.foextrap] * sd,
    )
    assert bcr.lo() == [amr.BCType.int_dir] * sd
    assert bcr.hi() == [amr.BCType.foextrap] * sd
    for d in range(sd):
        assert bcr.lo(d) == amr.BCType.int_dir
        assert bcr.hi(d) == amr.BCType.foextrap
    assert bcr.vect() == [amr.BCType.int_dir] * sd + [amr.BCType.foextrap] * sd
    assert bcr.data() == bcr.vect()


def test_bcrec_set():
    sd = amr.Config.spacedim
    bcr = amr.BCRec()
    for d in range(sd):
        bcr.set_lo(d, amr.BCType.reflect_even)
        bcr.set_hi(d, amr.BCType.ext_dir)
    assert bcr.lo() == [amr.BCType.reflect_even] * sd
    assert bcr.hi() == [amr.BCType.ext_dir] * sd


def test_bcrec_equality():
    sd = amr.Config.spacedim
    a = amr.BCRec(lo=[amr.BCType.int_dir] * sd, hi=[amr.BCType.int_dir] * sd)
    b = amr.BCRec(lo=[amr.BCType.int_dir] * sd, hi=[amr.BCType.int_dir] * sd)
    c = amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.int_dir] * sd)
    assert a == b
    assert a != c


def test_bcrec_from_box(std_box):
    """BCRec(bx, domain, bc_domain): inherits domain bndry types on the
    domain edge, interior Dirichlet elsewhere"""
    sd = amr.Config.spacedim
    bc_domain = amr.BCRec(
        lo=[amr.BCType.foextrap] * sd,
        hi=[amr.BCType.ext_dir] * sd,
    )

    # box at the low end of the domain
    lo_box = amr.Box(std_box.small_end, amr.IntVect(7))
    bcr = amr.BCRec(lo_box, std_box, bc_domain)
    assert bcr.lo() == [amr.BCType.foextrap] * sd
    assert bcr.hi() == [amr.BCType.int_dir] * sd

    # box in the interior of the domain
    in_box = amr.Box(amr.IntVect(8), amr.IntVect(15))
    bcr = amr.BCRec(in_box, std_box, bc_domain)
    assert bcr.lo() == [amr.BCType.int_dir] * sd
    assert bcr.hi() == [amr.BCType.int_dir] * sd


def test_set_bc(std_box):
    sd = amr.Config.spacedim
    bc_domain = amr.BCRec(
        lo=[amr.BCType.foextrap] * sd,
        hi=[amr.BCType.ext_dir] * sd,
    )
    lo_box = amr.Box(std_box.small_end, amr.IntVect(7))
    bcr = amr.setBC(lo_box, std_box, bc_domain)
    assert bcr.lo() == [amr.BCType.foextrap] * sd
    assert bcr.hi() == [amr.BCType.int_dir] * sd


def test_set_bc_vector(std_box):
    sd = amr.Config.spacedim
    ncomp = 3
    bc_domain = amr.Vector_BCRec(
        [
            amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.ext_dir] * sd)
            for _ in range(ncomp)
        ]
    )
    lo_box = amr.Box(std_box.small_end, amr.IntVect(7))
    bcr = amr.setBC(lo_box, std_box, 0, 0, ncomp, bc_domain)
    assert bcr.size() == ncomp
    for n in range(ncomp):
        assert bcr[n].lo() == [amr.BCType.foextrap] * sd
        assert bcr[n].hi() == [amr.BCType.int_dir] * sd


def test_set_bc_vector_with_offsets(std_box):
    sd = amr.Config.spacedim
    bc_domain = amr.Vector_BCRec(
        [
            amr.BCRec(lo=[amr.BCType.foextrap] * sd, hi=[amr.BCType.ext_dir] * sd),
            amr.BCRec(lo=[amr.BCType.hoextrap] * sd, hi=[amr.BCType.reflect_even] * sd),
            amr.BCRec(
                lo=[amr.BCType.reflect_odd] * sd, hi=[amr.BCType.ext_dir_cc] * sd
            ),
        ]
    )
    lo_box = amr.Box(std_box.small_end, amr.IntVect(7))
    bcr = amr.setBC(lo_box, std_box, 1, 1, 2, bc_domain)

    assert bcr.size() == 3
    assert bcr[0].lo() == [amr.BCType.bogus] * sd
    assert bcr[0].hi() == [amr.BCType.bogus] * sd
    assert bcr[1].lo() == [amr.BCType.hoextrap] * sd
    assert bcr[1].hi() == [amr.BCType.int_dir] * sd
    assert bcr[2].lo() == [amr.BCType.reflect_odd] * sd
    assert bcr[2].hi() == [amr.BCType.int_dir] * sd


def test_vector_bcrec():
    sd = amr.Config.spacedim
    bcr = amr.BCRec(lo=[amr.BCType.int_dir] * sd, hi=[amr.BCType.foextrap] * sd)
    v = amr.Vector_BCRec([bcr, bcr])
    assert v.size() == 2
    assert v[0] == bcr
    v[1] = amr.BCRec(lo=[amr.BCType.ext_dir] * sd, hi=[amr.BCType.ext_dir] * sd)
    assert v[1] != bcr
