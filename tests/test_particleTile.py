# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex


def test_ptile_pushback_ptiledata():

    pt = amrex.ParticleTile_1_1_2_1()
    p = amrex.Particle_1_1(1.,2.,3,4.,5)
    # p.set_rdata([4.])
    # p.set_idata([5])
    sp = amrex.Particle_3_2(5.,6.,7.,8.,9.,10.,11,12)
    # sp.set_rdata([8.,9.,10.])
    # sp.set_idata([11,12])
    pt.push_back(p)
    pt.push_back(sp)

    td = pt.getParticleTileData()

    assert(np.isclose(td[0].get_rdata(0),4.) and np.isclose(td[1].get_rdata(2), 10.) and td[1].get_idata(1) ==  12)

@pytest.mark.skipif(amrex.Config.spacedim != 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_access():
    pt = amrex.ParticleTile_1_1_2_1()
    sp1 = amrex.Particle_3_2()
    pt.push_back(sp1)
    pt.push_back(sp1)
    sp1.x = 2.
    sp1.z = 3.
    td = pt.getParticleTileData()
    td[0] = sp1
    sp2 = amrex.Particle_3_2()
    sp2.x = 5.
    sp2.y = 4.5
    pt[1] = sp2
    assert(np.isclose(pt[0].x, 2) and np.isclose(pt[0].y, 0) and np.isclose(pt[0].z, 3))
    assert(np.isclose(pt[1].x, 5) and np.isclose(pt[1].y, 4.5) and np.isclose(pt[1].z, 0))

def test_ptile_soa():
    pt = amrex.ParticleTile_1_1_2_1()

    pt.push_back_real(1, 2.1)
    pt.push_back_real([1.1,1.3])
    pt.push_back_real(1,3,3.14)


    pt.push_back_int(0, 10)
    pt.push_back_int([12])
    pt.push_back_int(0,3,31)

    rdata = pt.GetStructOfArrays().GetRealData()
    idata = pt.GetStructOfArrays().GetIntData()
    # print(rd[0].dtype)
    ar0 = np.array(rdata[0])
    ar1 = np.array(rdata[1])
    ir0 = np.array(idata[0])
    assert(np.allclose(ar0, np.array([1.1])))
    assert(np.allclose(ar1, np.array([2.1,1.3,3.14,3.14,3.14])))
    assert(np.allclose(ir0, np.array([10,12,31,31,31])))
    print(pt.GetStructOfArrays().GetIntData())

@pytest.mark.skipif(amrex.Config.spacedim != 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_ptile_aos():
    pt = amrex.ParticleTile_1_1_2_1()
    p1 = amrex.Particle_1_1()
    p2 = amrex.Particle_1_1()
    p1.x = 3.0
    p2.x = 4.0
    p2.y = 8
    pt.push_back(p1)

    pt.push_back(p2)
    p3 = amrex.Particle_1_1()
    p3.z = 20
    pt.push_back(p3)
    pt.resize(3)
    aos = np.array(pt.GetArrayOfStructs())
    print(aos)
    assert(np.isclose(aos[0]['x'], 3.0) and np.isclose(aos[0]['y'],0))
    assert(np.isclose(aos[2][0], 0.0) and np.isclose(aos[2][2], 20))


    #To test
    # push_back superParticle
    # push_back_real
    # shrink_to_fit
    # capacity
    # Num...Comps
    # swap
