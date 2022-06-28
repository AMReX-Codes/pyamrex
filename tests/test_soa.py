# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_soa_init():
    soa = amrex.StructOfArrays_2_1()
    soa.define(3,5)
    print(soa.size())

    print(soa.GetRealData(0))
    print(soa.GetIntData(1))
    print(soa.NumRealComps())
    print(soa.NumIntComps())
    # assert(False)

def test_something():
    soa = amrex.StructOfArrays_2_1()
    real_data = soa.GetRealData()
    print(real_data)

    # assert(False)





        # .def("define", &SOAType::define)
        # .def("NumRealComps", &SOAType::NumRealComps)
        # .def("NumIntComps", &SOAType::NumIntComps)
        # .def("GetRealData", [](SOAType& soa){ return soa.GetRealData();})
        # .def("GetRealData", [](SOAType& soa, const int index){ return soa.GetRealData(index);})
        # .def("GetIntData", [](SOAType& soa){ return soa.GetIntData();})
        # .def("GetIntData", [](SOAType& soa, const int index){ return soa.GetIntData(index);})
        # // .def("GetRealData", py::overload_cast<>(&SOAType::GetRealData))
        # // .def("GetRealData", py::overload_cast<>(&SOAType::GetRealData, py::const_))
        # // .def("GetRealData", py::overload_cast<int>(&SOAType::GetRealData))
        # // .def("GetRealData", py::overload_cast<int>(&SOAType::GetRealData, py::const_))
        # // .def("GetIntData", py::overload_cast<>(&SOAType::GetIntData))
        # // .def("GetIntData", py::overload_cast<>(&SOAType::GetIntData, py::const_))
        # .def("size", &SOAType::size)
        # .def("numParticles", &SOAType::numParticles)
        # .def("numRealParticles", &SOAType::numRealParticles)
        # .def("numTotalParticles", &SOAType::numTotalParticles)
        # .def("setNumNeighbors", &SOAType::setNumNeighbors)
        # .def("getNumNeighbors", &SOAType::getNumNeighbors)
        # .def("resize", &SOAType::resize)
