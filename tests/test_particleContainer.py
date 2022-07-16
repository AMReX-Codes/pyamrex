# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

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

    # pc.resizeData()
  
    lev = pc.GetParticles(0)
    for tile_ind, pt in lev.items():
        print('tile', tile_ind)
        real_arrays = pt.GetStructOfArrays().GetRealData()
        int_arrays = pt.GetStructOfArrays().GetIntData()
        aos = pt.GetArrayOfStructs()
        aos_arr = np.array(aos,copy=False)
        if len(real_arrays) > 0:
            assert(np.isclose(real_arrays[0][0], 0.5) and np.isclose(real_arrays[1][0],0.2))
            assert(isinstance(int_arrays[0][0], int))
            assert(int_arrays[0][0] == 1)
            assert(isinstance(aos_arr[0]['rdata_0'],np.floating))
            assert(isinstance(aos_arr[0]['idata_0'], np.integer))
            assert(np.isclose(aos_arr[0]['rdata_0'], 0.5) and aos_arr[0]['idata_0'] == 5)

            aos_arr[0]['idata_0'] = 2
            aos1 = pt.GetArrayOfStructs()
            print(aos1[0])
            print(aos[0])
            print(aos_arr[0])
            assert(aos_arr[0]['idata_0'] == aos[0].get_idata(0) == aos1[0].get_idata(0) == 2)

            print('soa test')
            real_arrays[1][0] = -1.2

            ra1 = pt.GetStructOfArrays().GetRealData()
            print(real_arrays)
            print(ra1)
            for ii, arr in enumerate(real_arrays):
                assert(np.allclose(np.array(arr), np.array(ra1[ii])))

            print('soa int test')
            iarr_np = np.array(int_arrays[0], copy=False)
            iarr_np[0] = -3
            ia1 = pt.GetStructOfArrays().GetIntData()
            ia1_np = np.array(ia1[0], copy = False)
            print(iarr_np)
            print(ia1_np)
            assert(np.allclose(iarr_np, ia1_np))


    print('---- is the particle tile recording changes or passed by reference? --------')
    lev1 = pc.GetParticles(0)
    for tile_ind, pt in lev1.items():
        print('tile', tile_ind)
        real_arrays = pt.GetStructOfArrays().GetRealData()
        int_arrays = pt.GetStructOfArrays().GetIntData()
        aos = pt.GetArrayOfStructs()
        print(aos[0])
        assert(aos[0].get_idata(0) == 2)
        assert(real_arrays[1][0] == -1.2)
        assert(int_arrays[0][0] == -3)
