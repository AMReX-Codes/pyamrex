# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def test_soa_init():
    soa = amr.StructOfArrays_3_1_default()
    print("--test init --")
    print("num real components", soa.num_real_comps)
    print("num int components", soa.num_int_comps)
    assert soa.num_real_comps == 3 and soa.num_int_comps == 1

    soa.define(1, 3, ["x", "y", "z", "w"], ["i1", "i2", "i3", "i4"])
    print("--test define --")
    print("num real components", soa.num_real_comps)
    print("num int components", soa.num_int_comps)
    assert soa.num_real_comps == 4 and soa.num_int_comps == 4
    print("num particles", soa.num_particles)
    print("num real particles", soa.num_real_particles)
    print("num totalparticles", soa.num_total_particles)
    print("num Neighbors", soa.get_num_neighbors())
    print("soa size", soa.size)
    assert soa.num_particles == soa.num_real_particles == 0
    assert soa.size == soa.num_total_particles == 0
    assert soa.get_num_neighbors() == 0

    soa.resize(5)
    print("--test resize --")
    print("num particles", soa.num_particles)
    print("num real particles", soa.num_real_particles)
    print("num totalparticles", soa.num_total_particles)
    print("num Neighbors", soa.get_num_neighbors())
    print("soa size", soa.size)
    assert soa.num_particles == soa.num_real_particles == 5
    assert soa.size == soa.num_total_particles == 5
    assert soa.get_num_neighbors() == 0

    soa.set_num_neighbors(3)
    print("--test set neighbor num--")
    print("num particles", soa.num_particles)
    print("num real particles", soa.num_real_particles)
    print("num totalparticles", soa.num_total_particles)
    print("num Neighbors", soa.get_num_neighbors())
    print("soa size", soa.size)
    assert soa.num_particles == soa.num_real_particles == 5
    assert soa.size == soa.num_total_particles == 8
    assert soa.get_num_neighbors() == 3


def test_soa_from_tile():
    pt = (
        amr.ParticleTile_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleTile_2_1_3_1_default()
    )
    p = amr.Particle_5_2(
        1.0,
        2.0,
        3.0,
        rdata_0=4.0,
        rdata_1=5.0,
        rdata_2=6.0,
        rdata_3=7.0,
        rdata_4=8.0,
        idata_0=9,
        idata_1=10,
    )
    sp = amr.Particle_5_2(1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9, 10)
    pt.push_back(p)
    pt.push_back(sp)

    soa = pt.get_struct_of_arrays()
    print("num particles", soa.num_particles)
    print("num real particles", soa.num_real_particles)
    print("num totalparticles", soa.num_total_particles)
    print("num Neighbors", soa.get_num_neighbors())
    print("soa size", soa.size)
    assert soa.num_particles == soa.size == 2
    assert soa.num_total_particles == soa.num_real_particles == 2
    assert soa.get_num_neighbors() == 0

    real_arrays = soa.get_real_data()
    int_arrays = soa.get_int_data()
    print(real_arrays)
    assert np.isclose(real_arrays[0][1], 6.1) and np.isclose(real_arrays[1][1], 7.1)
    assert isinstance(int_arrays[0][0], int)
    assert int_arrays[0][1] == 10

    real_arrays[1][0] = -1.2

    ra1 = soa.get_real_data()
    print(real_arrays)
    print(ra1)
    for ii, arr in enumerate(real_arrays):
        assert np.allclose(arr.to_numpy(), ra1[ii].to_numpy())

    print("soa int test")
    iarr_np = int_arrays[0].to_numpy()
    iarr_np[0] = -3
    ia1 = soa.get_int_data()
    ia1_np = ia1[0].to_numpy()
    print(iarr_np)
    print(ia1_np)
    assert np.allclose(iarr_np, ia1_np)
