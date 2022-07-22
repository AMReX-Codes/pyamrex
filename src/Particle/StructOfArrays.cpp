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
        .def("GetRealData", py::overload_cast<>(&SOAType::GetRealData),
            py::return_value_policy::reference_internal)
        .def("GetIntData", py::overload_cast<>(&SOAType::GetIntData),
            py::return_value_policy::reference_internal)
        .def("size", &SOAType::size)
        .def("numParticles", &SOAType::numParticles)
        .def("numRealParticles", &SOAType::numRealParticles)
        .def("numTotalParticles", &SOAType::numTotalParticles)
        .def("setNumNeighbors", &SOAType::setNumNeighbors)
        .def("getNumNeighbors", &SOAType::getNumNeighbors)
        .def("resize", &SOAType::resize)
    ;
}


void init_StructOfArrays(py::module& m) {
    make_StructOfArrays< 2, 1 > (m);
    make_StructOfArrays< 4, 0 > (m);  // HiPACE++ 22.07
    make_StructOfArrays< 5, 0 > (m);  // ImpactX 22.07
    make_StructOfArrays< 7, 0 > (m);
    make_StructOfArrays< 37, 1> (m);  // HiPACE++ 22.07
}
