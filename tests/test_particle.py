# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_particle_init():
    p1 = amrex.Particle_7_0()
    nreal = len(amrex.PIdx.RealValues.__members__)
    nint = len(amrex.PIdx.IntValues.__members__)
    print('particle',p1)
    assert(amrex.Particle_7_0.NReal == nreal)
    assert(amrex.Particle_7_0.NInt == nint)
    assert(p1.NReal == nreal)
    assert(p1.NInt == nint)

@pytest.mark.skipif(amrex.Config.spacedim != 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_particle_set():
    p1 = amrex.Particle_7_0()
    p1.setPos(1,1.5)
    assert(p1.pos(0) == 0 and p1.pos(1) == 1.5 and p1.pos(2) == 0)
    p1.setPos([1.,1,2])
    assert(p1.pos(0) == 1 and p1.pos(1) == 1 and p1.pos(2) == 2)
    p1.setPos(amrex.RealVect(2,3.3,4.2))
    assert(p1.pos(0) == 2 and p1.pos(1) == 3.3 and p1.pos(2) == 4.2)

    print(p1.x, p1.y, p1.z)
    p1.x = 2.1
    assert(p1.x == 2.1)
    p1.y = 3.2
    assert(p1.y == 3.2)
    p1.z = 5.1
    assert(p1.z == 5.1)

def test_id_cpu():
    p1 = amrex.Particle_2_1()
    print(p1.cpu() )
    print(p1.id() )
    p2 = amrex.Particle_2_1()
    print(p1.cpu() )
    print(p1.id() )
    # assert(False)

def test_nextid():
    p1 = amrex.Particle_2_1()
    print(p1.id())
    # print(amrex.Particle.the_next_id)
    print(p1.NextID())

    p2 = amrex.Particle_2_1()
    print(p1.id())
    print(p2.id())
    print(p1.NextID())

    p1.NextID(12)
    print(p1.NextID())
    # print(amrex.Particle.the_next_id)
    # assert(False)

def test_rdata():
    p1 = amrex.Particle_2_1()
    rvec = [1.5,2.0]
    p1.set_rdata(rvec)
    print(p1)
    assert(p1.get_rdata() == rvec)
    p1.set_rdata(1, 2.5)
    print(p1.get_rdata())
    assert(p1.get_rdata(1)==2.5)
    test_passed = False
    try:
        p1.set_rdata(100,5.2)
    except ValueError:
        test_passed = True
    assert(test_passed)

    test_passed = False
    try:
        p1.get_rdata(100)
    except ValueError:
        test_passed = True
    assert(test_passed)

def test_idata():
    p1 = amrex.Particle_2_1()
    ivec = [-1]
    p1.set_idata(ivec)
    print(p1)
    assert(p1.get_idata() == ivec)
    p1.set_idata(0, 3)
    print(p1.get_idata())
    assert(p1.get_idata(0)==3)
    test_passed = False
    try:
        p1.set_idata(100,5)
    except ValueError:
        test_passed = True
    assert(test_passed)

    test_passed = False
    try:
        p1.get_idata(100)
    except ValueError:
        test_passed = True
    assert(test_passed)