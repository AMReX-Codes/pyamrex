/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_StructOfArrays.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


template <int NReal, int NInt,
          template<class> class Allocator=DefaultAllocator>
void make_StructOfArrays(py::module &m)
{
    using SOAType = StructOfArrays<NReal, NInt>;

    auto const soa_name = std::string("StructOfArrays_").append(std::to_string(NReal) + "_" + std::to_string(NInt));
    py::class_<SOAType>(m, soa_name.c_str())
        .def(py::init())
        .def("define", &SOAType::define)
        .def("NumRealComps", &SOAType::NumRealComps)
        .def("NumIntComps", &SOAType::NumIntComps)
        .def("GetRealData", [](SOAType& soa){ return soa.GetRealData();})
        .def("GetRealData", [](SOAType& soa, const int index){ return soa.GetRealData(index);})
        .def("GetIntData", [](SOAType& soa){ return soa.GetIntData();})
        .def("GetIntData", [](SOAType& soa, const int index){ return soa.GetIntData(index);})
        // .def("GetRealData", py::overload_cast<>(&SOAType::GetRealData))
        // .def("GetRealData", py::overload_cast<>(&SOAType::GetRealData, py::const_))
        // .def("GetRealData", py::overload_cast<int>(&SOAType::GetRealData))
        // .def("GetRealData", py::overload_cast<int>(&SOAType::GetRealData, py::const_))
        // .def("GetIntData", py::overload_cast<>(&SOAType::GetIntData))
        // .def("GetIntData", py::overload_cast<>(&SOAType::GetIntData, py::const_))
        .def("size", &SOAType::size)
        .def("numParticles", &SOAType::numParticles)
        .def("numRealParticles", &SOAType::numRealParticles)
        .def("numTotalParticles", &SOAType::numTotalParticles)
        .def("setNumNeighbors", &SOAType::setNumNeighbors)
        .def("getNumNeighbors", &SOAType::getNumNeighbors)
        .def("resize", &SOAType::resize)
        // .def("__array_interface__", )
    ;
}


void init_StructOfArrays(py::module& m) {
    make_StructOfArrays< 2, 1> (m);
}