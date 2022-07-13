# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

#TODO : Tests struct of arrays

@pytest.fixture()
def particle_container(std_geometry, distmap, boxarr):
    pc = amrex.ParticleContainer_1_1_2_1(std_geometry, distmap, boxarr)
    return pc

def test_n_particles(particle_container):
    pc = particle_container
    assert(pc.OK())
    assert(pc.NStructReal == amrex.ParticleContainer_1_1_2_1.NStructReal == 1)
    assert(pc.NStructInt == amrex.ParticleContainer_1_1_2_1.NStructInt == 1)
    assert(pc.NArrayReal == amrex.ParticleContainer_1_1_2_1.NArrayReal == 2)
    assert(pc.NArrayInt == amrex.ParticleContainer_1_1_2_1.NArrayInt == 1)


def test_pc_init():
    pc = amrex.ParticleContainer_1_1_2_1()


    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0,0,0,1,1,1)
    coord_int = 1 # RZ
    periodicity = [0,0,1]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    pc.Define(gm,dm,ba)

    amrex.ParticleContainer_1_1_2_1(gm,dm,ba)

def test_particle_init(particle_container, std_real_box):
    pc = particle_container

    myt = amrex.ParticleInitType_1_1_2_1()
    myt.real_struct_data = [0.5]
    myt.int_struct_data = [5]
    myt.real_array_data = [0.5, 0.2]
    myt.int_array_data = [1]

    Npart = 21
    iseed = 1
    pc.InitRandom(Npart,iseed,myt,False,std_real_box)
    assert(pc.NumberOfParticlesAtLevel(0) == np.sum(pc.NumberOfParticlesInGrid(0))==Npart)

    pc.resizeData()

    pt = pc.DefineAndReturnParticleTile(0,0,0)


    aos = np.array(pt.GetArrayOfStructs())
    rdata = pt.GetStructOfArrays().GetRealData()
    idata = pt.GetStructOfArrays().GetIntData()

    lev = pc.GetParticles(0)
    for key in lev.keys():
        pt = lev[key]
        real_arr = pt.GetStructOfArrays().GetRealData()
        int_arr = pt.GetStructOfArrays().GetIntData()
        aos = np.array(pt.GetArrayOfStructs())
        # print(real_arr.size)
        if len(real_arr) > 0:
            # print(aos[1].x)
            # print(len(real_arr))
            # print(aos.__array_interface__)
            assert(np.isclose(real_arr[0][0], 0.5) and np.isclose(real_arr[1][0],0.2) and np.isclose(aos[0][3], 0.5) and np.isclose(aos[0][5], 5))
