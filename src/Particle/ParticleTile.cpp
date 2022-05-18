/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParticleTile.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

//Forward declaration
template <int T_NReal, int T_NInt=0>
void make_Particle(py::module &m);

template <int NStructReal, int NStructInt, int NArrayReal, int NArrayInt,
          template<class> class Allocator=DefaultAllocator>
void make_ParticleTile(py::module &m)
{

    // make_Particle<NArrayReal, NArrayInt> (m);

    using ParticleTileDataType = ParticleTileData<NStructReal, NStructInt, NArrayReal, NArrayInt>;
    using ParticleTileType=ParticleTile<NStructReal, NStructInt, NArrayReal, NArrayInt, DefaultAllocator>;
    using ParticleType = Particle<NStructReal, NStructInt>;
    static constexpr int NAR = NArrayReal;
    static constexpr int NAI = NArrayInt;

    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    py::class_<ParticleTileDataType>(m, "ParticleTileData")
        .def(py::init())
        .def_readonly("m_size", &ParticleTileDataType::m_size)
        .def_readonly("m_num_runtime_real", &ParticleTileDataType::m_num_runtime_real)
        .def_readonly("m_num_runtime_int", &ParticleTileDataType::m_num_runtime_int)
        .def("getSuperParticle", &ParticleTileDataType::getSuperParticle)
        .def("setSuperParticle", &ParticleTileDataType::setSuperParticle)
    ;

    py::class_<ParticleTileType>(m, "ParticleTile")
        .def(py::init())
        .def_readonly_static("NAR", &ParticleTileType::NAR)
        .def_readonly_static("NAI", &ParticleTileType::NAI)
        .def("define", &ParticleTileType::define)
        // .def("GetArrayOfStructs")
        // .def("GetStructOfArrays")
        .def("empty", &ParticleTileType::empty)
        .def("size", &ParticleTileType::size)
        .def("numParticles", &ParticleTileType::numParticles)
        .def("numRealParticles", &ParticleTileType::numRealParticles)
        .def("numNeighborParticles", &ParticleTileType::numNeighborParticles)
        .def("numTotalParticles", &ParticleTileType::numTotalParticles)
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
    ;
}

void init_ParticleTile(py::module& m) {
    make_ParticleTile< 1, 1, 2, 1> (m);
}