/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_GpuAllocators.H>
#include <AMReX_StructOfArrays.H>

#include <sstream>


template <int NReal, int NInt,
          template<class> class Allocator=amrex::DefaultAllocator,
          bool use64BitIdCpu=false>
void make_StructOfArrays(py::module &m, std::string allocstr)
{
    using namespace amrex;

    using SOAType = StructOfArrays<NReal, NInt, Allocator, use64BitIdCpu>;

    auto soa_name = std::string("StructOfArrays_") + std::to_string(NReal) + "_" +
                    std::to_string(NInt);
    if (use64BitIdCpu)
        soa_name += "_idcpu";
    soa_name += "_" + allocstr;

    py::class_<SOAType> py_SoA(m, soa_name.c_str());
    py_SoA
        .def(py::init())
        .def("define", &SOAType::define)
        .def_property_readonly("num_real_comps", &SOAType::NumRealComps,
             "Get the number of compile-time + runtime Real components")
        .def_property_readonly("num_int_comps", &SOAType::NumIntComps,
             "Get the number of compile-time + runtime Int components")
        .def_property_readonly("has_idcpu", [](const py::object&){ return use64BitIdCpu; },
             "In pure SoA particle layout, idcpu is an array in the SoA")

        // compile-time components
        .def("get_real_data", py::overload_cast<>(&SOAType::GetRealData),
            py::return_value_policy::reference_internal,
            "Get access to the particle Real Arrays (only compile-time components)")
        .def("get_int_data", py::overload_cast<>(&SOAType::GetIntData),
            py::return_value_policy::reference_internal,
            "Get access to the particle Int Arrays (only compile-time components)")
        // compile-time and runtime components
        .def("get_real_data", py::overload_cast<const int>(&SOAType::GetRealData),
             py::return_value_policy::reference_internal,
             py::arg("index"),
             "Get access to a particle Real component Array (compile-time and runtime component)")
        .def("get_int_data", py::overload_cast<const int>(&SOAType::GetIntData),
             py::return_value_policy::reference_internal,
             py::arg("index"),
             "Get access to a particle Real component Array (compile-time and runtime component)")

        .def("__len__", &SOAType::size,
             "Get the number of particles")
        .def_property_readonly("size", &SOAType::size,
             "Get the number of particles")
        .def_property_readonly("num_particles", &SOAType::numParticles)
        .def_property_readonly("num_real_particles", &SOAType::numRealParticles)
        .def_property_readonly("num_total_particles", &SOAType::numTotalParticles)

        .def("set_num_neighbors", &SOAType::setNumNeighbors)
        .def("get_num_neighbors", &SOAType::getNumNeighbors)
        .def("resize", &SOAType::resize)
    ;
    if (use64BitIdCpu)
        py_SoA.def("get_idcpu_data", py::overload_cast<>(&SOAType::GetIdCPUData),
                   py::return_value_policy::reference_internal,
                   "Get access to a particle IdCPU component Array");
}

template <int NReal, int NInt, bool use64BitIdCpu=false>
void make_StructOfArrays(py::module &m)
{
    // first, because used as copy target in methods in containers with other allocators
    make_StructOfArrays<NReal, NInt, amrex::PinnedArenaAllocator, use64BitIdCpu>(m, "pinned");

    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/pybind11/pull/4581
    //make_StructOfArrays<NReal, NInt, std::allocator, use64BitIdCpu>(m, "std");
    //make_StructOfArrays<NReal, NInt, amrex::ArenaAllocator, use64BitIdCpu>(m, "arena");
#ifdef AMREX_USE_GPU
    make_StructOfArrays<NReal, NInt, std::allocator, use64BitIdCpu>(m, "std");
    make_StructOfArrays<NReal, NInt, amrex::DefaultAllocator, use64BitIdCpu> (m, "default");  // amrex::ArenaAllocator
#else
    make_StructOfArrays<NReal, NInt, amrex::DefaultAllocator, use64BitIdCpu> (m, "default");  // std::allocator
    make_StructOfArrays<NReal, NInt, amrex::ArenaAllocator, use64BitIdCpu>(m, "arena");
#endif
    //   end work-around
#ifdef AMREX_USE_GPU
    make_StructOfArrays<NReal, NInt, amrex::DeviceArenaAllocator, use64BitIdCpu>(m, "device");
    make_StructOfArrays<NReal, NInt, amrex::ManagedArenaAllocator, use64BitIdCpu>(m, "managed");
    make_StructOfArrays<NReal, NInt, amrex::AsyncArenaAllocator, use64BitIdCpu>(m, "async");
#endif
}

void init_StructOfArrays(py::module& m) {
    make_StructOfArrays< 2, 1>(m);
    make_StructOfArrays< 4, 0>(m);  // HiPACE++ 22.08 - 24.02
    make_StructOfArrays< 5, 0>(m);  // ImpactX 22.07 - 24.02
#if AMREX_SPACEDIM == 1
    make_StructOfArrays< 5, 0, true>(m);  // WarpX 24.03+ 1D
#elif AMREX_SPACEDIM == 2
    make_StructOfArrays< 6, 0, true>(m);  // WarpX 24.03+ 2D
    make_StructOfArrays< 7, 0, true>(m);  // WarpX 24.03+ RZ
#elif AMREX_SPACEDIM == 3
    make_StructOfArrays< 7, 0, true>(m);  // WarpX 24.03+ 3D
#endif
    make_StructOfArrays< 8, 0, true>(m);  // ImpactX 24.03+
    make_StructOfArrays<37, 1>(m);  // HiPACE++ 22.09 - 24.02
}
