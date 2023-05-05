/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_StructOfArrays.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <sstream>

namespace py = nanobind;
using namespace amrex;


template <int NReal, int NInt,
          template<class> class Allocator=DefaultAllocator>
void make_StructOfArrays(py::module_ &m, std::string allocstr)
{
    using SOAType = StructOfArrays<NReal, NInt, Allocator>;

    auto const soa_name = std::string("StructOfArrays_") + std::to_string(NReal) + "_" +
                          std::to_string(NInt) + "_" + allocstr;
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

template <int NReal, int NInt>
void make_StructOfArrays(py::module_ &m)
{
    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/nanobind/pull/4581
    //make_StructOfArrays<NReal, NInt, std::allocator>(m, "std");
    //make_StructOfArrays<NReal, NInt, amrex::ArenaAllocator>(m, "arena");
#ifdef AMREX_USE_GPU
    make_StructOfArrays<NReal, NInt, std::allocator>(m, "std");
    make_StructOfArrays<NReal, NInt, amrex::DefaultAllocator> (m, "default");  // amrex::ArenaAllocator
#else
    make_StructOfArrays<NReal, NInt, amrex::DefaultAllocator> (m, "default");  // std::allocator
    make_StructOfArrays<NReal, NInt, amrex::ArenaAllocator>(m, "arena");
#endif
    //   end work-around
    make_StructOfArrays<NReal, NInt, amrex::PinnedArenaAllocator>(m, "pinned");
#ifdef AMREX_USE_GPU
    make_StructOfArrays<NReal, NInt, amrex::DeviceArenaAllocator>(m, "device");
    make_StructOfArrays<NReal, NInt, amrex::ManagedArenaAllocator>(m, "managed");
    make_StructOfArrays<NReal, NInt, amrex::AsyncArenaAllocator>(m, "async");
#endif
}

void init_StructOfArrays(py::module_& m) {
    make_StructOfArrays< 2, 1>(m);
    make_StructOfArrays< 4, 0>(m);  // HiPACE++ 22.07
    make_StructOfArrays< 5, 0>(m);  // ImpactX 22.07 - 23.06
    make_StructOfArrays< 8, 2>(m);  // ImpactX 23.06+
    make_StructOfArrays<37, 1>(m);  // HiPACE++ 22.07
}
