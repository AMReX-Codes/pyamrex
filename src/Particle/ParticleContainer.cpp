/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
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

void init_ParticleContainer(py::module& m)
{
    using ParticleContainerType = ParticleContainer<0,0,6,2>;
    py::class_<ParticleContainerType>(m, "ParticleContainer")
    .def(py::init())
    .def(py::init<const Geometry&, const DistributionMapping&, const BoxArray&>())


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

    // .def("DefineAndReturnParticleTile",
    //     py::overload_cast<int, int, int>
    //     (&ParticleContainerType::DefineAndReturnParticleTile))
    ;
}