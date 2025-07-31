# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


##########
def test_ptile_data():
    ptd = amr.ParticleTileData_2_1_3_1()
    assert ptd.m_size == 0
    assert ptd.m_num_runtime_real == 0
    assert ptd.m_num_runtime_int == 0


def test_ptile_funs():
    pt = amr.ParticleTile_2_1_3_1_default()

    assert pt.empty and pt.size == 0
    assert pt.num_particles == pt.num_real_particles == pt.num_neighbor_particles == 0
    assert pt.num_total_particles == pt.get_num_neighbors() == 0

    pt.set_num_neighbors(3)
    assert pt.num_particles == 0 == pt.num_real_particles
    assert pt.num_neighbor_particles == pt.get_num_neighbors() == 3
    assert pt.num_total_particles == 3

    pt.resize(5)
    assert not pt.empty and pt.size == 5
    assert pt.num_particles == 2 == pt.num_real_particles
    assert pt.num_neighbor_particles == pt.get_num_neighbors() == 3
    assert pt.num_total_particles == 5


################
def test_ptile_pushback_ptiledata():
    pt = (
        amr.ParticleTile_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleTile_2_1_3_1_default()
    )
    p = amr.Particle_2_1(1.0, 2.0, 3.0, 4.0, 5.0, 6)
    sp = amr.Particle_5_2(5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 11, 12)
    pt.push_back(p)
    pt.push_back(sp)

    print("num particles", pt.num_particles)
    print("num real particles", pt.num_real_particles)
    print("num neighbor particles", pt.num_neighbor_particles)
    print("num totalparticles", pt.num_total_particles)
    print("num Neighbors", pt.get_num_neighbors())
    print("tile is empty?", pt.empty)
    print("tile size", pt.size)
    assert not pt.empty
    assert pt.num_particles == pt.num_real_particles == pt.num_total_particles == 2
    assert pt.num_neighbor_particles == pt.get_num_neighbors() == 0

    td = pt.get_particle_tile_data()
    assert (
        np.isclose(td[0].get_rdata(0), 4.0)
        and np.isclose(td[1].get_rdata(2), 10.0)
        and td[1].get_idata(1) == 12
    )


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_access():
    pt = (
        amr.ParticleTile_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleTile_2_1_3_1_default()
    )
    sp1 = amr.Particle_5_2()
    pt.push_back(sp1)
    pt.push_back(sp1)
    sp1.x = 2.0
    sp1.z = 3.0
    td = pt.get_particle_tile_data()
    td[0] = sp1
    sp2 = amr.Particle_5_2()
    sp2.x = 5.0
    sp2.y = 4.5
    pt[1] = sp2
    assert np.isclose(pt[0].x, 2) and np.isclose(pt[0].y, 0) and np.isclose(pt[0].z, 3)
    assert (
        np.isclose(pt[1].x, 5) and np.isclose(pt[1].y, 4.5) and np.isclose(pt[1].z, 0)
    )


def test_ptile_soa():
    pt = (
        amr.ParticleTile_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleTile_2_1_3_1_default()
    )

    pt.push_back_real(1, 2, 2.1)
    pt.push_back_real([1.1, 1.3, 1.5])
    pt.push_back_real(1, 3, 3.14)

    pt.push_back_int(0, 10)
    pt.push_back_int([12])
    pt.push_back_int(0, 3, 31)

    rdata = pt.get_struct_of_arrays().get_real_data()
    idata = pt.get_struct_of_arrays().get_int_data()
    print("rdata: ", rdata)

    ar0 = np.array(rdata[0], copy=False)
    ar1 = np.array(rdata[1], copy=False)
    ir0 = np.array(idata[0], copy=False)
    print(ar0.dtype)
    assert ar0.dtype == "float" or ar0.dtype == "float32"
    assert ir0.dtype == "int32"
    print("---------")
    ir0[0] = -55
    print(ir0)
    print(idata)
    assert ir0[0] == idata[0][0] == -55
    print("-------")
    idata[0][0] = -66
    print(ir0)
    print(idata)
    assert ir0[0] == idata[0][0] == -66
    print("-------")
    # np.array(rdata[0])
    assert np.allclose(ar0, np.array([1.1]))
    assert np.allclose(ar1, np.array([2.1, 2.1, 1.3, 3.14, 3.14, 3.14]))
    assert np.allclose(ir0, np.array([-66, 12, 31, 31, 31]))


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_aos_3d():
    pt = (
        amr.ParticleTile_2_1_3_1_managed()
        if amr.Config.have_gpu
        else amr.ParticleTile_2_1_3_1_default()
    )
    p1 = amr.Particle_2_1()
    p2 = amr.Particle_2_1()
    p1.x = 3.0
    p2.x = 4.0
    p2.y = 8
    pt.push_back(p1)

    pt.push_back(p2)
    p3 = amr.Particle_2_1()
    p3.z = 20
    pt.push_back(p3)
    pt.resize(3)
    aos = np.array(pt.get_array_of_structs())
    print(aos)
    assert np.isclose(aos[0]["x"], 3.0) and np.isclose(aos[0]["y"], 0)
    assert np.isclose(aos[2][0], 0.0) and np.isclose(aos[2][2], 20)


def test_ptile_aos():
    idcpu = np.array(
        [
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
        ],
        dtype=np.uint64,
    )
    ids = amr.unpack_ids(idcpu)
    cpus = amr.unpack_cpus(idcpu)
    assert np.array_equal(ids, np.array([1, 1, 1, 1, 1]))
    assert np.array_equal(cpus, np.array([100, 100, 100, 100, 100]))

    assert amr.is_valid(idcpu[0])
    idcpu[0] = amr.make_invalid(idcpu[0])
    assert not amr.is_valid(idcpu[0])
    idcpu[0] = amr.make_valid(idcpu[0])
    assert amr.is_valid(idcpu[0])

    # the leftmost bit stores the sign of the id
    # the next 39 store its absolute value
    # the rightmost 24 then store the cpu number
    # using this scheme, id = 1 cpu = 100
    # corresponds to 9223372036871553124
    assert idcpu[0] == 9223372036871553124

    idcpu = np.array([0, 0, 0, 0, 0], dtype=np.uint64)
    amr.pack_ids(idcpu, np.array([1, 1, 1, 1, 1], dtype=np.int64))
    amr.pack_cpus(idcpu, np.array([100, 100, 100, 100, 100], dtype=np.int32))
    print(idcpu)
    assert np.array_equal(
        idcpu,
        np.array(
            [
                9223372036871553124,
                9223372036871553124,
                9223372036871553124,
                9223372036871553124,
                9223372036871553124,
            ]
        ),
    )

    idcpu = np.array(
        [
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
            9223372036871553124,
        ]
    )
    ids = amr.unpack_ids(idcpu)
    cpus = amr.unpack_cpus(idcpu)
    assert np.array_equal(ids, np.array([1, 1, 1, 1, 1]))
    assert np.array_equal(cpus, np.array([100, 100, 100, 100, 100]))
