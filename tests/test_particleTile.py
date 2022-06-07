# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_ptile_data():
    
    ptd = amrex.ParticleTileData_1_1_2_1()
    sp = amrex.Particle_3_2()
    # sp.setPos([1,2,3])
    sp.x = 1
    # ptd.setSuperParticle(sp,0)
    # ptd.getSuperParticle(0)
    print(ptd.m_size)
    print(ptd.m_num_runtime_real)
    print(ptd.m_num_runtime_int)
    
    # assert(False)


def test_init_ptile():
    pt = amrex.ParticleTile_1_1_2_1()

    print(pt.empty())
    print(pt.size())

    pt.define(4,3)

    print(pt.empty())
    print(pt.size())
    # assert(False)

def test_ptile_funs():

    pt = amrex.ParticleTile_1_1_2_1()
    print('num particles', pt.numParticles())
    print('num real particles', pt.numRealParticles())
    print('num neighbor particles', pt.numNeighborParticles())
    print('num totalparticles', pt.numTotalParticles())
    print('num Neighbors', pt.getNumNeighbors())
    pt.setNumNeighbors(3)
    print('num Neighbors', pt.getNumNeighbors())
    pt.resize(5)
    print('tile is empty?', pt.empty())
    print('tile size', pt.size())
    # assert(False)

def test_ptile_pushback():

    pt = amrex.ParticleTile_1_1_2_1()
    p = amrex.Particle_1_1()
    sp = amrex.Particle_3_2()
    pt.push_back(p)
    pt.push_back(sp)

    print('num particles', pt.numParticles())
    print('num real particles', pt.numRealParticles())
    print('num neighbor particles', pt.numNeighborParticles())
    print('num totalparticles', pt.numTotalParticles())
    print('num Neighbors', pt.getNumNeighbors())
    print('num Neighbors', pt.getNumNeighbors())
    print('tile is empty?', pt.empty())
    print('tile size', pt.size())

    pt.push_back_real(1, 2.1)
    pt.push_back_real([1.1,1.3])
    pt.push_back_real(1,3,3.14)


    pt.push_back_int(0, 10)
    pt.push_back_int([12])
    pt.push_back_int(0,3,31)

    td = pt.getParticleTileData()
    print('particle tile data size', td.m_size)
    # for ii in range(pt.size()):
    for ii in range(7):
        print('particle',ii)
        print(td[ii])
    assert(td[2].get_rdata(1)==1.1 and td[2].get_rdata(2)==2.1 and td[2].get_idata(1)==10)

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
    print('------')
    print(pt[0])
    print(pt[1])
    assert(pt[0].x==2 and pt[0].y==0 and pt[0].z==3)
    assert(pt[1].x==5 and pt[1].y==4.5 and pt[1].z==0)

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
    print(rdata)
    print(rdata[0])
    print(rdata[0].__array_interface__)
    # print(rd[0].dtype)
    ar0 = np.array(rdata[0])
    ar1 = np.array(rdata[1])
    ir0 = np.array(idata[0])
    assert(ar0 == np.array([1.1]))
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
    assert(aos[0]['x']==3.0 and aos[0]['y']==0)
    assert(aos[2][0]==0.0 and aos[2][2] == 20)


    #To test
    # push_back superParticle
    # push_back_real
    # shrink_to_fit
    # capacity
    # Num...Comps
    # swap