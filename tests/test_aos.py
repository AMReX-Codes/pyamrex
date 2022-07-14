# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

# TODO: test numNeighborParticles

def test_aos_init():
    aos = amrex.ArrayOfStructs_2_1()
    # print(aos())

    assert(aos.numParticles() == 0)
    assert(aos.numParticles() == aos.numTotalParticles() == aos.numRealParticles()==0)
    assert(aos.empty())

def test_aos_push_pop():
    aos = amrex.ArrayOfStructs_2_1()
    p1 = amrex.Particle_2_1()
    p1.set_rdata([1.5, 2.2])
    p1.set_idata([3])
    aos.push_back(p1)
    p2 = amrex.Particle_2_1()
    p2.set_rdata([2.1, 25.2])
    p2.set_idata([5])
    aos.push_back(p2)

    assert(aos.numParticles() == aos.numTotalParticles() == aos.numRealParticles()==2)
    assert(aos.getNumNeighbors() == 0)
    aos.setNumNeighbors(5)
    assert(aos.getNumNeighbors() == 5)
    assert(not aos.empty())
    assert(aos.size()==7)
    assert(aos[0].get_rdata() == p1.get_rdata())
    p3 = amrex.Particle_2_1()
    p3.set_rdata([3.14, -3.14])
    p3.set_idata([10])
    aos[0] = p3
    assert(aos[0].get_idata() == p3.get_idata())
    

    aos.pop_back()
    assert(aos.numParticles() == aos.numRealParticles() == 1)
    assert(aos.numTotalParticles() == 6)

def test_array_interface():
    aos = amrex.ArrayOfStructs_2_1()
    p1 = amrex.Particle_2_1()
    p1.setPos([1,2,3])
    p1.set_rdata([4.5, 5.2])
    p1.set_idata([6])
    aos.push_back(p1)
    p2 = amrex.Particle_2_1()
    p2.setPos([8,9,10])
    p2.set_rdata([11.1, 12.2])
    p2.set_idata([13])
    aos.push_back(p2)

    print(aos[0])
    print(dir(aos[0]))

    # print('particle 1 from aos:\n',aos[0])
    # print('particle 2 from aos:\n',aos[1])
    # print('array interface\n', aos.__array_interface__)
    arr = np.array(aos, copy=False)
    assert(np.isclose(arr[0][0],1.0) and np.isclose(arr[0][4],5.2) and np.isclose(arr[0][6], 6))
    assert(np.isclose(arr[1][2],10) and np.isclose(arr[1][3],11.1) and np.isclose(arr[1][6], 13))

    p3 = amrex.Particle_2_1(x=-3)
    p4 = amrex.Particle_2_1(y=-5)
    print(arr)
    print(aos[0],aos[1])
    print('-------')
    aos[0] = p3
    print('array:',arr)
    print('aos[0]:',aos[0],'aos[1]:',aos[1])
    assert(aos[0].x==arr[0][0]==-3)
    assert(aos[0].y==arr[0][1]==0)
    assert(aos[0].z==arr[0][2]==0)
    
    shape = amrex.Config.spacedim + amrex.Particle_2_1.NReal + amrex.Particle_2_1.NInt + 1
    for ii in range(shape):
        arr[1][ii] = 0
    arr[1][1] = -5# np.array([0, -5, 0,0,0,0,0])
    print('array:',arr)
    print('aos[0]:',aos[0],'aos[1]:',aos[1])
    assert(aos[1].y==arr[1][1]==-5)
    assert(aos[1].x==arr[1][0]==0)
    assert(aos[1].z==arr[1][2]==0)