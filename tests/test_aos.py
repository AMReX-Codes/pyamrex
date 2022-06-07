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

    print(aos[0])
    print('size',aos.size())
    print('numPart', aos.numParticles())
    print('numRealPart', aos.numRealParticles())
    print('numNeighborPart', aos.numNeighborParticles())
    print('numTotalPart', aos.numTotalParticles())
    assert(aos.numParticles() == aos.numTotalParticles() == aos.numRealParticles()==2)
    print('getNumNeighbors',aos.getNumNeighbors())
    assert(aos.getNumNeighbors() == 0)
    aos.setNumNeighbors(5)
    assert(aos.getNumNeighbors() == 5)
    print('aos.getNumNeighbors()', aos.getNumNeighbors())
    assert(not aos.empty())
    print('empty', aos.empty())
        # //data, dataPtr
        # .def("push_back", &AOSType::push_back)
        # .def("pop_back", &AOSType::pop_back)
        # .def("back", py::overload_cast<>(&AOSType::back))
        # .def("back", py::overload_cast<>(&AOSType::back, py::const_))
        # // setter & getter
        # .def("__setitem__", [](AOSType &aos, int const v,  const ParticleType& p){ aos[v] = p; })
        # .def("__getitem__", [](AOSType &aos, int const v){ return aos[v]; })
    assert(aos.size()==7)
    print(aos[0].get_rdata())
    assert(aos[0].get_rdata() == p1.get_rdata())
    p3 = amrex.Particle_2_1()
    p3.set_rdata([3.14, -3.14])
    p3.set_idata([10])
    aos[0] = p3
    assert(aos[0].get_idata() == p3.get_idata())
    
    print(aos.back())

    aos.pop_back()
    print(aos.size())
    print('numPart', aos.numParticles())
    print('numRealPart', aos.numRealParticles())
    print('numTotalPart', aos.numTotalParticles())
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


    print('particle 1 from aos:\n',aos[0])
    print('particle 2 from aos:\n',aos[1])
    print('array interface\n', aos.__array_interface__)
    arr = np.array(aos)
    print('numpy array of aos\n', arr)
    assert(arr[0][0]==1.0 and arr[0][4]==5.2 and arr[0][6]==6)
    assert(arr[1][2]==10 and arr[1][3]==11.1 and arr[1][6]==13)