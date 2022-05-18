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
#include <AMReX_Particles.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

// impactx particle container uses:
// SetParticleSize (protected)
// reserveData
// reesizeData
// DefineAndReturnParticleTile
// copyParticles


// template<int NStructReal, int NStructInt, int NArrayReal, int NArrayInt>
// void make_ParticleInitType(py::module &m)
// {
// }

template <int T_NStructReal, int T_NStructInt=0, int T_NArrayReal=0, int T_NArrayInt=0,
          template<class> class Allocator=DefaultAllocator>
void make_ParticleContainer(py::module &m)
{
    using ParticleInitData = ParticleInitType<T_NStructReal,T_NStructInt,T_NArrayReal,T_NArrayInt>;
    py::class_<ParticleInitData>(m, "ParticleInitType")
        .def(py::init<>())
        .def_readwrite("real_struct_data", &ParticleInitData::real_struct_data)
        .def_readwrite("int_struct_data", &ParticleInitData::int_struct_data)
        .def_readwrite("real_array_data", &ParticleInitData::real_array_data)
        .def_readwrite("int_array_data", &ParticleInitData::int_array_data)
    ;

    using ParticleContainerType = ParticleContainer<T_NStructReal,T_NStructInt,T_NArrayReal,T_NArrayInt,Allocator>;
    py::class_<ParticleContainerType>(m, "ParticleContainer")
        .def(py::init())
        .def(py::init<const Geometry&, const DistributionMapping&, const BoxArray&>())

        // .def_propery_readonly_static("NStructReal", &ParticleContainerType::NStructReal)
        .def_property_readonly_static("NStructReal", [](const py::object&){return ParticleContainerType::NStructReal; })
        .def_property_readonly_static("NStructInt", [](const py::object&){return ParticleContainerType::NStructInt; })
        .def_property_readonly_static("NArrayReal", [](const py::object&){return ParticleContainerType::NArrayReal; })
        .def_property_readonly_static("NArrayInt", [](const py::object&){return ParticleContainerType::NArrayInt; })


        // ParticleContainer (const Vector<Geometry>            & geom,
        //                    const Vector<DistributionMapping> & dmap,
        //                    const Vector<BoxArray>            & ba,
        //                    const Vector<int>                 & rr)

        // ParticleContainer (const Vector<Geometry>            & geom,
        //                    const Vector<DistributionMapping> & dmap,
        //                    const Vector<BoxArray>            & ba,
        //                    const Vector<IntVect>             & rr)


        // ParticleContainer ( const ParticleContainer &) = delete;
        // ParticleContainer& operator= ( const ParticleContainer & ) = delete;

        // ParticleContainer ( ParticleContainer && ) = default;
        // ParticleContainer& operator= ( ParticleContainer && ) = default;

        .def("Define",
                py::overload_cast<const Geometry&,
                                    const DistributionMapping&,
                                    const BoxArray&>
                (&ParticleContainerType::Define))

        // void Define (const Vector<Geometry>            & geom,
        //              const Vector<DistributionMapping> & dmap,
        //              const Vector<BoxArray>            & ba,
        //              const Vector<int>                 & rr)

        //   void Define (const Vector<Geometry>            & geom,
        //              const Vector<DistributionMapping> & dmap,
        //              const Vector<BoxArray>            & ba,
        //              const Vector<IntVect>             & rr)

        // int numLocalTilesAtLevel (int lev) const { return m_particles[lev].size(); }
        .def("numLocalTilesAtLevel", &ParticleContainerType::numLocalTilesAtLevel)

        // void reserveData ();
        .def("reserveData", &ParticleContainerType::reserveData)
        // void resizeData ();
        .def("resizeData", &ParticleContainerType::resizeData)

        // void InitFromAsciiFile (const std::string& file, int extradata,
        //                         const IntVect* Nrep = nullptr);

        // void InitFromBinaryFile (const std::string& file, int extradata);

        // void InitFromBinaryMetaFile
        //     void InitRandom (Long icount, ULong iseed,
        //                  const ParticleInitData& mass,
        //                  bool serialize = false, RealBox bx = RealBox());

        .def("InitRandom", py::overload_cast<Long, ULong, const ParticleInitData&, bool, RealBox>(&ParticleContainerType::InitRandom))

        .def("InitRandomPerBox", py::overload_cast<Long, ULong, const ParticleInitData&>(&ParticleContainerType::InitRandomPerBox))

        // .def("DefineAndReturnParticleTile",
        //     py::overload_cast<int, int, int>
        //     (&ParticleContainerType::DefineAndReturnParticleTile))
    ;
}


void init_ParticleContainer(py::module& m) {
    make_ParticleContainer< 1, 1, 2, 1> (m);
}