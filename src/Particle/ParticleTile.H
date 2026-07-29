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
void make_ParticleTileData(nb::module_ &m)
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

    nb::class_<ParticleTileDataType>(m, particle_tile_data_type.c_str())
            .def(nb::init())

            .def_prop_ro("m_size", [](ParticleTileDataType const & ptd){ return ptd.m_size; })
            .def_prop_ro("m_num_runtime_real", [](ParticleTileDataType const & ptd){ return ptd.m_num_runtime_real; })
            .def_prop_ro("m_num_runtime_int", [](ParticleTileDataType const & ptd){ return ptd.m_num_runtime_int; })

            .def("get_super_particle", [](ParticleTileDataType &pdt, int const v) -> SuperParticleType { return pdt.getSuperParticle(v); })
            .def("set_super_particle", [](ParticleTileDataType &pdt, SuperParticleType const& sp, int const v) { pdt.setSuperParticle(sp, v); })
            // setter & getter
            .def("__setitem__", [](ParticleTileDataType &pdt, int const v,
                                   SuperParticleType const value) { pdt.setSuperParticle(value, v); })
            .def("__getitem__",
                 [](ParticleTileDataType &pdt, int const v) { return pdt.getSuperParticle(v); })

    ;
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator=amrex::DefaultAllocator>
void make_ParticleTile(nb::module_ &m, std::string allocstr)
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

    auto py_particle_tile = nb::class_<ParticleTileType>(m, particle_tile_type.c_str())
        .def(nb::init())
        .def_ro_static("NAR", &ParticleTileType::NAR)
        .def_ro_static("NAI", &ParticleTileType::NAI)
        .def("define", &ParticleTileType::define)
        .def("get_struct_of_arrays", nb::overload_cast<>(&ParticleTileType::GetStructOfArrays),
            nb::rv_policy::reference_internal)

        .def_prop_ro("empty", &ParticleTileType::empty)
        .def_prop_ro("size", &ParticleTileType::size)
        .def_prop_ro("num_particles", &ParticleTileType::numParticles)
        .def_prop_ro("num_real_particles", &ParticleTileType::numRealParticles)
        .def_prop_ro("num_neighbor_particles", &ParticleTileType::numNeighborParticles)
        .def_prop_ro("num_total_particles", &ParticleTileType::numTotalParticles)

        .def("set_num_neighbors", &ParticleTileType::setNumNeighbors)
        .def("get_num_neighbors", &ParticleTileType::getNumNeighbors)
        .def("resize",
            &ParticleTileType::resize,
            nb::arg("count"),
            nb::arg("strategy") = GrowthStrategy::Poisson
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

        .def_prop_ro("num_real_comps", &ParticleTileType::NumRealComps)
        .def_prop_ro("num_int_comps", &ParticleTileType::NumIntComps)
        .def_prop_ro("num_runtime_real_comps", &ParticleTileType::NumRuntimeRealComps)
        .def_prop_ro("num_runtime_int_comps", &ParticleTileType::NumRuntimeIntComps)

        .def("shrink_to_fit", &ParticleTileType::shrink_to_fit)
        .def("capacity", &ParticleTileType::capacity)
        .def("swap",&ParticleTileType::swap)
        .def("get_particle_tile_data", &ParticleTileType::getParticleTileData,
             nb::keep_alive<0, 1>())
        .def("__setitem__", [](ParticleTileType & pt, int const v, SuperParticleType const value){ pt.getParticleTileData().setSuperParticle( value, v); })
        .def("__getitem__", [](ParticleTileType & pt, int const v){ return pt.getParticleTileData().getSuperParticle(v); })
    ;

    if constexpr (!T_ParticleType::is_soa_particle) {
        py_particle_tile
            .def("get_array_of_structs",
                 nb::overload_cast<>(&ParticleTileType::GetArrayOfStructs),
                 nb::rv_policy::reference_internal)
        ;
    }
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt, bool only_polymorphic=false>
void make_ParticleTile(nb::module_ &m)
{
    if constexpr (T_ParticleType::is_soa_particle) {
        make_ParticleTileData<amrex::SoAParticleBase, NArrayReal, NArrayInt>(m);
    }
    else {
        make_ParticleTileData<T_ParticleType, NArrayReal, NArrayInt>(m);
    }

    if constexpr (!only_polymorphic) {
        // first, because used as copy target in methods in containers with other allocators
        make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                          amrex::PinnedArenaAllocator>(m, "pinned");

        // see Src/Base/AMReX_GpuContainers.H
        //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
        //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

        // Register each concrete allocator type only once: DefaultAllocator is
        // an alias of std::allocator on CPU and ArenaAllocator on GPU.
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
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::PolymorphicArenaAllocator>(m, "polymorphic");
}
