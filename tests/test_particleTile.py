# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


##########
def test_ptile_data():
    ptd = amr.ParticleTileData_1_1_2_1()
    assert ptd.m_size == 0
    assert ptd.m_num_runtime_real == 0
    assert ptd.m_num_runtime_int == 0


def test_ptile_funs():
    pt = amr.ParticleTile_1_1_2_1_default()

    assert pt.empty() and pt.size() == 0
    assert pt.numParticles() == pt.numRealParticles() == pt.numNeighborParticles() == 0
    assert pt.numTotalParticles() == pt.getNumNeighbors() == 0

    pt.setNumNeighbors(3)
    assert pt.numParticles() == 0 == pt.numRealParticles()
    assert pt.numNeighborParticles() == pt.getNumNeighbors() == 3
    assert pt.numTotalParticles() == 3

    pt.resize(5)
    assert not pt.empty() and pt.size() == 5
    assert pt.numParticles() == 2 == pt.numRealParticles()
    assert pt.numNeighborParticles() == pt.getNumNeighbors() == 3
    assert pt.numTotalParticles() == 5


################
def test_ptile_pushback_ptiledata():
    pt = amr.ParticleTile_1_1_2_1_default()
    p = amr.Particle_1_1(1.0, 2.0, 3, 4.0, 5)
    sp = amr.Particle_3_2(5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11, 12)
    pt.push_back(p)
    pt.push_back(sp)

    print("num particles", pt.numParticles())
    print("num real particles", pt.numRealParticles())
    print("num neighbor particles", pt.numNeighborParticles())
    print("num totalparticles", pt.numTotalParticles())
    print("num Neighbors", pt.getNumNeighbors())
    print("tile is empty?", pt.empty())
    print("tile size", pt.size())
    assert not pt.empty()
    assert pt.numParticles() == pt.numRealParticles() == pt.numTotalParticles() == 2
    assert pt.numNeighborParticles() == pt.getNumNeighbors() == 0

    td = pt.getParticleTileData()
    assert (
        np.isclose(td[0].get_rdata(0), 4.0)
        and np.isclose(td[1].get_rdata(2), 10.0)
        and td[1].get_idata(1) == 12
    )


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_access():
    pt = amr.ParticleTile_1_1_2_1_default()
    sp1 = amr.Particle_3_2()
    pt.push_back(sp1)
    pt.push_back(sp1)
    sp1.x = 2.0
    sp1.z = 3.0
    td = pt.getParticleTileData()
    td[0] = sp1
    sp2 = amr.Particle_3_2()
    sp2.x = 5.0
    sp2.y = 4.5
    pt[1] = sp2
    assert np.isclose(pt[0].x, 2) and np.isclose(pt[0].y, 0) and np.isclose(pt[0].z, 3)
    assert (
        np.isclose(pt[1].x, 5) and np.isclose(pt[1].y, 4.5) and np.isclose(pt[1].z, 0)
    )


def test_ptile_soa():
    pt = amr.ParticleTile_1_1_2_1_default()

    pt.push_back_real(1, 2.1)
    pt.push_back_real([1.1, 1.3])
    pt.push_back_real(1, 3, 3.14)

    pt.push_back_int(0, 10)
    pt.push_back_int([12])
    pt.push_back_int(0, 3, 31)

    rdata = pt.GetStructOfArrays().GetRealData()
    idata = pt.GetStructOfArrays().GetIntData()
    print("rdata: ", rdata)

    ar0 = np.array(rdata[0], copy=False)
    ar1 = np.array(rdata[1], copy=False)
    ir0 = np.array(idata[0], copy=False)
    print(ar0.dtype)
    assert ar0.dtype == "float"
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
    assert np.allclose(ar1, np.array([2.1, 1.3, 3.14, 3.14, 3.14]))
    assert np.allclose(ir0, np.array([-66, 12, 31, 31, 31]))


@pytest.mark.skipif(amr.Config.spacedim != 3, reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_aos():
    pt = amr.ParticleTile_1_1_2_1_default()
    p1 = amr.Particle_1_1()
    p2 = amr.Particle_1_1()
    p1.x = 3.0
    p2.x = 4.0
    p2.y = 8
    pt.push_back(p1)

    pt.push_back(p2)
    p3 = amr.Particle_1_1()
    p3.z = 20
    pt.push_back(p3)
    pt.resize(3)
    aos = np.array(pt.GetArrayOfStructs())
    print(aos)
    assert np.isclose(aos[0]["x"], 3.0) and np.isclose(aos[0]["y"], 0)
    assert np.isclose(aos[2][0], 0.0) and np.isclose(aos[2][2], 20)
