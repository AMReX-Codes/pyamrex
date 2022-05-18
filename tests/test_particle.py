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

def test_particle_set():
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
    assert(False)

def test_nextid():
    p1 = amrex.Particle()
    print(p1.id())
    # print(amrex.Particle.the_next_id)
    print(p1.NextID())

    p2 = amrex.Particle()
    print(p1.id())
    print(p2.id())
    print(p1.NextID())

    p1.NextID(12)
    print(p1.NextID())
    # print(amrex.Particle.the_next_id)
    assert(False)