# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def test_soa_init():
    soa = amr.StructOfArrays_2_1_default()
    print("--test init --")
    print("num real components", soa.num_real_comps)
    print("num int components", soa.num_int_comps)
    assert soa.num_real_comps == 2 and soa.num_int_comps == 1

    soa.define(1, 3)
    print("--test define --")
    print("num real components", soa.num_real_comps)
    print("num int components", soa.num_int_comps)
    assert soa.num_real_comps == 3 and soa.num_int_comps == 4
    print("num particles", soa.numParticles())
    print("num real particles", soa.numRealParticles())
    print("num totalparticles", soa.numTotalParticles())
    print("num Neighbors", soa.getNumNeighbors())
    print("soa size", soa.size())
    assert soa.numParticles() == soa.numRealParticles() == 0
    assert soa.size() == soa.numTotalParticles() == 0
    assert soa.getNumNeighbors() == 0

    soa.resize(5)
    print("--test resize --")
    print("num particles", soa.numParticles())
    print("num real particles", soa.numRealParticles())
    print("num totalparticles", soa.numTotalParticles())
    print("num Neighbors", soa.getNumNeighbors())
    print("soa size", soa.size())
    assert soa.numParticles() == soa.numRealParticles() == 5
    assert soa.size() == soa.numTotalParticles() == 5
    assert soa.getNumNeighbors() == 0

    soa.setNumNeighbors(3)
    print("--test set neighbor num--")
    print("num particles", soa.numParticles())
    print("num real particles", soa.numRealParticles())
    print("num totalparticles", soa.numTotalParticles())
    print("num Neighbors", soa.getNumNeighbors())
    print("soa size", soa.size())
    assert soa.numParticles() == soa.numRealParticles() == 5
    assert soa.size() == soa.numTotalParticles() == 8
    assert soa.getNumNeighbors() == 3


def test_soa_from_tile():
    pt = amr.ParticleTile_1_1_2_1_default()
    p = amr.Particle_1_1(1.0, 2.0, 3, rdata_0=4.0, idata_1=5)
    sp = amr.Particle_3_2(
        5.0, 6.0, 7.0, rdata_0=8.0, rdata_1=9.0, rdata_2=10.0, idata_0=11, idata_1=12
    )
    pt.push_back(p)
    pt.push_back(sp)

    soa = pt.GetStructOfArrays()
    print("num particles", soa.numParticles())
    print("num real particles", soa.numRealParticles())
    print("num totalparticles", soa.numTotalParticles())
    print("num Neighbors", soa.getNumNeighbors())
    print("soa size", soa.size())
    assert soa.numParticles() == soa.size() == 2
    assert soa.numTotalParticles() == soa.numRealParticles() == 2
    assert soa.getNumNeighbors() == 0

    real_arrays = soa.GetRealData()
    int_arrays = soa.GetIntData()
    print(real_arrays)
    assert np.isclose(real_arrays[0][1], 9) and np.isclose(real_arrays[1][1], 10)
    assert isinstance(int_arrays[0][0], int)
    assert int_arrays[0][1] == 12

    real_arrays[1][0] = -1.2

    ra1 = soa.GetRealData()
    print(real_arrays)
    print(ra1)
    for ii, arr in enumerate(real_arrays):
        assert np.allclose(arr.to_numpy(), ra1[ii].to_numpy())

    print("soa int test")
    iarr_np = int_arrays[0].to_numpy()
    iarr_np[0] = -3
    ia1 = soa.GetIntData()
    ia1_np = ia1[0].to_numpy()
    print(iarr_np)
    print(ia1_np)
    assert np.allclose(iarr_np, ia1_np)
