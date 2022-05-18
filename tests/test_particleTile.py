# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_ptile_data():
    
    ptd = amrex.ParticleTileData()
    sp = amrex.Particle_3_2()
    # sp.setPos([1,2,3])
    sp.x = 1
    # ptd.setSuperParticle(sp,0)
    # ptd.getSuperParticle(0)
    print(ptd.m_size)
    print(ptd.m_num_runtime_real)
    print(ptd.m_num_runtime_int)
    
    assert(False)


def test_init_ptile():
    pt = amrex.ParticleTile()

    print(pt.empty())
    print(pt.size())

    pt.define(4,3)

    print(pt.empty())
    print(pt.size())
    assert(False)

def test_ptile_funs():

    pt = amrex.ParticleTile()
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
    assert(False)

def test_ptile_pushback():

    pt = amrex.ParticleTile()
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


    pt.push_back_real(1, 10)
    pt.push_back_real([11,12])
    pt.push_back_real(1,3,31)

    td = pt.getParticleTileData()
    print(td.m_size)
    assert(False)

    #To test
    # push_back superParticle
    # push_back_real
    # shrink_to_fit
    # capacity
    # Num...Comps
    # swap