/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Particle.H>

#include <sstream>


template <typename T_ParticleType, int NArrayReal, int NArrayInt>
void make_ParticleTileData(py::module &m)
{
    using namespace amrex;

    using ParticleType = T_ParticleType;
    constexpr int NStructReal = ParticleType::NReal;
    constexpr int NStructInt = ParticleType::NInt;

    using ParticleTileDataType = ParticleTileData<T_ParticleType, NArrayReal, NArrayInt>;
    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    auto particle_tile_data_type = std::string("ParticleTileData_");
    if (T_ParticleType::is_soa_particle) {
        particle_tile_data_type += "pureSoA_";
    }
    else {
        particle_tile_data_type +=
            std::to_string(NStructReal) + "_" +
            std::to_string(NStructInt) + "_";
    }
    particle_tile_data_type +=
        std::to_string(NArrayReal) + "_" +
        std::to_string(NArrayInt);

    py::class_<ParticleTileDataType>(m, particle_tile_data_type.c_str())
            .def(py::init())

            .def_property_readonly("m_size", [](ParticleTileDataType const & ptd){ return ptd.m_size; })
            .def_property_readonly("m_num_runtime_real", [](ParticleTileDataType const & ptd){ return ptd.m_num_runtime_real; })
            .def_property_readonly("m_num_runtime_int", [](ParticleTileDataType const & ptd){ return ptd.m_num_runtime_int; })

            .def("get_super_particle", &ParticleTileDataType::template getSuperParticle<ParticleType>)
            .def("set_super_particle", &ParticleTileDataType::template setSuperParticle<ParticleType>)
            // setter & getter
            .def("__setitem__", [](ParticleTileDataType &pdt, int const v,
                                   SuperParticleType const value) { pdt.setSuperParticle(value, v); })
            .def("__getitem__",
                 [](ParticleTileDataType &pdt, int const v) { return pdt.getSuperParticle(v); })

    ;
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator=amrex::DefaultAllocator>
void make_ParticleTile(py::module &m, std::string allocstr)
{
    using namespace amrex;

    using ParticleType = T_ParticleType;
    constexpr int NStructReal = ParticleType::NReal;
    constexpr int NStructInt = ParticleType::NInt;

    using ParticleTileType = ParticleTile<T_ParticleType, NArrayReal, NArrayInt, Allocator>;
    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    auto particle_tile_type = std::string("ParticleTile_");
    if (T_ParticleType::is_soa_particle) {
        particle_tile_type += "pureSoA_";
    }
    else {
        particle_tile_type +=
            std::to_string(NStructReal) + "_" +
            std::to_string(NStructInt) + "_";
    }
    particle_tile_type += std::to_string(NArrayReal) + "_" +
                          std::to_string(NArrayInt) + "_" + allocstr;

    auto py_particle_tile = py::class_<ParticleTileType>(m, particle_tile_type.c_str())
        .def(py::init())
        .def_readonly_static("NAR", &ParticleTileType::NAR)
        .def_readonly_static("NAI", &ParticleTileType::NAI)
        .def("define", &ParticleTileType::define)
        .def("get_struct_of_arrays", py::overload_cast<>(&ParticleTileType::GetStructOfArrays),
            py::return_value_policy::reference_internal)

        .def_property_readonly("empty", &ParticleTileType::empty)
        .def_property_readonly("size", &ParticleTileType::size)
        .def_property_readonly("num_particles", &ParticleTileType::numParticles)
        .def_property_readonly("num_real_particles", &ParticleTileType::numRealParticles)
        .def_property_readonly("num_neighbor_particles", &ParticleTileType::numNeighborParticles)
        .def_property_readonly("num_total_particles", &ParticleTileType::numTotalParticles)

        .def("set_num_neighbors", &ParticleTileType::setNumNeighbors)
        .def("get_num_neighbors", &ParticleTileType::getNumNeighbors)
        .def("resize",
            &ParticleTileType::resize,
            py::arg("count"),
            py::arg("strategy") = GrowthStrategy::Poisson
        )
    ;

    if constexpr (!T_ParticleType::is_soa_particle) {
        py_particle_tile
            .def("push_back",
                 [](ParticleTileType& ptile, const ParticleType &p) { ptile.push_back(p); },
                 "Add one particle to this tile.")
        ;
    }

    py_particle_tile
        .def("push_back",
             [](ParticleTileType& ptile, const SuperParticleType &p) {ptile.push_back(p);},
             "Add one particle to this tile.")
        .def("push_back_real", [](ParticleTileType& ptile, int comp, ParticleReal v) {ptile.push_back_real(comp, v);})
        .def("push_back_real", [](ParticleTileType& ptile,
                            const std::array<ParticleReal, NArrayReal>& v) {ptile.push_back_real(v);})
        .def("push_back_real", [](ParticleTileType&ptile,
                                int comp,
                                std::size_t npar,
                                ParticleReal v)
                                {ptile.push_back_real(comp, npar, v);})
        .def("push_back_int", [](ParticleTileType& ptile, int comp, int v) {ptile.push_back_int(comp, v);})
        .def("push_back_int", [](ParticleTileType& ptile,
                                const std::array<int, NArrayInt>& v) {ptile.push_back_int(v);})
        .def("push_back_int", [](ParticleTileType&ptile,
                                int comp,
                                std::size_t npar,
                                int v)
                                {ptile.push_back_int(comp, npar, v);})

        .def_property_readonly("num_real_comps", &ParticleTileType::NumRealComps)
        .def_property_readonly("num_int_comps", &ParticleTileType::NumIntComps)
        .def_property_readonly("num_runtime_real_comps", &ParticleTileType::NumRuntimeRealComps)
        .def_property_readonly("num_runtime_int_comps", &ParticleTileType::NumRuntimeIntComps)

        .def("shrink_to_fit", &ParticleTileType::shrink_to_fit)
        .def("capacity", &ParticleTileType::capacity)
        .def("swap",&ParticleTileType::swap)
        .def("get_particle_tile_data", &ParticleTileType::getParticleTileData)
        .def("__setitem__", [](ParticleTileType & pt, int const v, SuperParticleType const value){ pt.getParticleTileData().setSuperParticle( value, v); })
        .def("__getitem__", [](ParticleTileType & pt, int const v){ return pt.getParticleTileData().getSuperParticle(v); })
    ;

    if constexpr (!T_ParticleType::is_soa_particle) {
        py_particle_tile
            .def("get_array_of_structs",
                 py::overload_cast<>(&ParticleTileType::GetArrayOfStructs),
                 py::return_value_policy::reference_internal)
        ;
    }
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
void make_ParticleTile(py::module &m)
{
    if constexpr (T_ParticleType::is_soa_particle) {
        make_ParticleTileData<amrex::SoAParticleBase, NArrayReal, NArrayInt>(m);
    }
    else {
        make_ParticleTileData<T_ParticleType, NArrayReal, NArrayInt>(m);
    }

    // first, because used as copy target in methods in containers with other allocators
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::PinnedArenaAllocator>(m, "pinned");

    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/pybind11/pull/4581
    //make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
    //                  std::allocator>(m, "std");
    //make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
    //                  amrex::ArenaAllocator>(m, "arena");
#ifdef AMREX_USE_GPU
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      std::allocator>(m, "std");
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::DefaultAllocator>(m, "default");  // amrex::ArenaAllocator
#else
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::DefaultAllocator>(m, "default");  // std::allocator
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::ArenaAllocator>(m, "arena");
#endif
    //   end work-around
#ifdef AMREX_USE_GPU
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::DeviceArenaAllocator>(m, "device");
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::ManagedArenaAllocator>(m, "managed");
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::AsyncArenaAllocator>(m, "async");
#endif
}
