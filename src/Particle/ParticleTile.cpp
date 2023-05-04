/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParticleTile.H>

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

//Forward declaration
template <int T_NReal, int T_NInt=0>
void make_Particle(py::module_ &m);

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
void make_ParticleTileData(py::module_ &m) {
    using ParticleType = T_ParticleType;
    constexpr int NStructReal = ParticleType::NReal;
    constexpr int NStructInt = ParticleType::NInt;

    using ParticleTileDataType = ParticleTileData<T_ParticleType, NArrayReal, NArrayInt>;
    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    auto const particle_tile_data_type =
        std::string("ParticleTileData_") + std::to_string(NStructReal) + "_" +
        std::to_string(NStructInt) + "_" + std::to_string(NArrayReal) + "_" +
        std::to_string(NArrayInt);
    py::class_<ParticleTileDataType>(m, particle_tile_data_type.c_str())
            .def(py::init())
            .def_ro("m_size", &ParticleTileDataType::m_size)
            .def_ro("m_num_runtime_real", &ParticleTileDataType::m_num_runtime_real)
            .def_ro("m_num_runtime_int", &ParticleTileDataType::m_num_runtime_int)
            .def("getSuperParticle", &ParticleTileDataType::template getSuperParticle<ParticleType>)
            .def("setSuperParticle", &ParticleTileDataType::setSuperParticle)
            // setter & getter
            .def("__setitem__", [](ParticleTileDataType &pdt, int const v,
                                   SuperParticleType const value) { pdt.setSuperParticle(value, v); })
            .def("__getitem__",
                 [](ParticleTileDataType &pdt, int const v) { return pdt.getSuperParticle(v); })

    ;
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator=DefaultAllocator>
void make_ParticleTile(py::module_ &m, std::string allocstr)
{
    using ParticleType = T_ParticleType;
    constexpr int NStructReal = ParticleType::NReal;
    constexpr int NStructInt = ParticleType::NInt;

    using ParticleTileType = ParticleTile<T_ParticleType, NArrayReal, NArrayInt, Allocator>;
    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    auto const particle_tile_type = std::string("ParticleTile_") + std::to_string(NStructReal) + "_" +
                                    std::to_string(NStructInt) + "_" + std::to_string(NArrayReal) + "_" +
                                    std::to_string(NArrayInt) + "_" + allocstr;
    py::class_<ParticleTileType>(m, particle_tile_type.c_str())
        .def(py::init())
        .def_ro_static("NAR", &ParticleTileType::NAR)
        .def_ro_static("NAI", &ParticleTileType::NAI)
        .def("define", &ParticleTileType::define)
        .def("GetArrayOfStructs",
            py::overload_cast<>(&ParticleTileType::GetArrayOfStructs),
            py::return_value_policy::reference_internal)
        .def("GetStructOfArrays", py::overload_cast<>(&ParticleTileType::GetStructOfArrays),
            py::return_value_policy::reference_internal)
        .def("empty", &ParticleTileType::empty)
        .def("size", &ParticleTileType::template size<ParticleType>)
        .def("numParticles", &ParticleTileType::template numParticles<ParticleType>)
        .def("numRealParticles", &ParticleTileType::template numRealParticles<ParticleType>)
        .def("numNeighborParticles", &ParticleTileType::template numNeighborParticles<ParticleType>)
        .def("numTotalParticles", &ParticleTileType::template numTotalParticles<ParticleType>)
        .def("setNumNeighbors", &ParticleTileType::setNumNeighbors)
        .def("getNumNeighbors", &ParticleTileType::getNumNeighbors)
        .def("resize", &ParticleTileType::resize)

        .def("push_back", [](ParticleTileType& ptile, const ParticleType &p){ ptile.push_back(p);})
        // .def("push_back", py::overload_cast<const ParticleType&>(&ParticleTileType::push_back), "Add one particle to this tile.")
        // .def("push_back", py::overload_cast<const SuperParticleType&>(&ParticleTileType::push_back), "Add one particle to this tile.")

        .def("push_back", [](ParticleTileType& ptile, const SuperParticleType &p) {ptile.push_back(p);})
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
        .def("NumRealComps", &ParticleTileType::NumRealComps)
        .def("NumIntComps", &ParticleTileType::NumIntComps)
        .def("NumRuntimeRealComps", &ParticleTileType::NumRuntimeRealComps)
        .def("NumRuntimeIntComps", &ParticleTileType::NumRuntimeIntComps)
        .def("shrink_to_fit", &ParticleTileType::shrink_to_fit)
        .def("capacity", &ParticleTileType::capacity)
        .def("swap",&ParticleTileType::swap)
        .def("getParticleTileData", &ParticleTileType::getParticleTileData)
        .def("__setitem__", [](ParticleTileType & pt, int const v, SuperParticleType const value){ pt.getParticleTileData().setSuperParticle( value, v); })
        .def("__getitem__", [](ParticleTileType & pt, int const v){ return pt.getParticleTileData().getSuperParticle(v); })
    ;
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
void make_ParticleTile(py::module_ &m)
{
    make_ParticleTileData<T_ParticleType, NArrayReal, NArrayInt>(m);

    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/nanobind/pull/4581
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
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::PinnedArenaAllocator>(m, "pinned");
#ifdef AMREX_USE_GPU
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::DeviceArenaAllocator>(m, "device");
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::ManagedArenaAllocator>(m, "managed");
    make_ParticleTile<T_ParticleType, NArrayReal, NArrayInt,
                      amrex::AsyncArenaAllocator>(m, "async");
#endif
}

void init_ParticleTile(py::module_& m) {
    // AMReX legacy AoS position + id/cpu particle ype
    using ParticleType_0_0 = Particle<0, 0>;
    using ParticleType_1_1 = Particle<1, 1>;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleTile<ParticleType_1_1, 2, 1> (m);
    make_ParticleTile<ParticleType_0_0, 4, 0> (m);   // HiPACE++ 22.07
    make_ParticleTile<ParticleType_0_0, 5, 0> (m);   // ImpactX 22.07
    make_ParticleTile<ParticleType_0_0, 7, 0> (m);
    make_ParticleTile<ParticleType_0_0, 37, 1> (m);  // HiPACE++ 22.07
}
