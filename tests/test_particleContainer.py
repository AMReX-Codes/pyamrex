# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

@pytest.fixture()
def particle_container(std_geometry, distmap, boxarr):
    pc = amrex.ParticleContainer(std_geometry, distmap, boxarr)
    return pc

def test_n_particles(particle_container):
    pc = particle_container
    print(pc.NStructReal)
    print(pc.NStructInt)
    print(pc.NArrayReal)
    print(pc.NArrayInt)
    print(amrex.ParticleContainer.NStructReal)
    print(amrex.ParticleContainer.NStructInt)
    print(amrex.ParticleContainer.NArrayReal)
    print(amrex.ParticleContainer.NArrayInt)
    assert(False)

def test_particleInitType():
    myt = amrex.ParticleInitType()
    print(myt.real_struct_data)
    print(myt.int_struct_data)
    print(myt.real_array_data)
    print(myt.int_array_data)

    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]
    print(myt.real_struct_data)
    print(myt.int_struct_data)
    print(myt.real_array_data)
    print(myt.int_array_data)
    assert(False)

def test_pc_init():
    pc = amrex.ParticleContainer()


    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0,0,0,1,1,1)
    coord_int = 1 # RZ
    periodicity = [0,0,1]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    pc.Define(gm,dm,ba)

    amrex.ParticleContainer(gm,dm,ba)

    assert(False)

def test_particle_init(particle_container, std_real_box):
    pc = particle_container

    myt = amrex.ParticleInitType()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]

    pc.InitRandomPerBox(5,0,myt)

    pc.InitRandom(5,0,myt,False,std_real_box)


def test_ptile():

    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0,0,0,1,1,1)
    coord_int = 1 # RZ
    periodicity = [0,0,0]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    pc = amrex.ParticleContainer(gm,dm,ba)

    print(pc.numLocalTilesAtLevel(0))
    assert(False) # replace with suitable test for numLocalTilesAtLevel

    pc.DefineAndReturnParticleTile()
