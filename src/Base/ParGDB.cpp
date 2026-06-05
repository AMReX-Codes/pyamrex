/* Copyright 2024-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParGDB.H>

#ifdef AMREX_PARTICLES
#   include <AMReX_AmrCore.H>
#   include <AMReX_AmrParGDB.H>
#endif


void init_ParGDB (py::module& m)
{
    using namespace amrex;

    // Abstract base: the Geometry/DistributionMapping/BoxArray broker handed to
    // a ParticleContainer. No Python constructor (it is pure virtual); concrete
    // instances come from AmrParGDB or are built C++-side.
    py::class_< ParGDBBase >(m, "ParGDBBase")
        .def("particle_geom",
             py::overload_cast< int >(&ParGDBBase::ParticleGeom, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))
        .def("geom",
             py::overload_cast< int >(&ParGDBBase::Geom, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))
        .def("particle_dist_map",
             py::overload_cast< int >(&ParGDBBase::ParticleDistributionMap, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))
        .def("dist_map",
             py::overload_cast< int >(&ParGDBBase::DistributionMap, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))
        .def("particle_box_array",
             py::overload_cast< int >(&ParGDBBase::ParticleBoxArray, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))
        .def("box_array",
             py::overload_cast< int >(&ParGDBBase::boxArray, py::const_),
             py::return_value_policy::reference_internal, py::arg("level"))

        .def("set_particle_box_array", &ParGDBBase::SetParticleBoxArray,
             py::arg("level"), py::arg("new_ba"))
        .def("set_particle_dist_map", &ParGDBBase::SetParticleDistributionMap,
             py::arg("level"), py::arg("new_dm"))
        .def("set_particle_geometry", &ParGDBBase::SetParticleGeometry,
             py::arg("level"), py::arg("new_geom"))

        .def("level_defined", &ParGDBBase::LevelDefined, py::arg("level"))
        .def("finest_level", &ParGDBBase::finestLevel)
        .def("max_level", &ParGDBBase::maxLevel)
        .def("ref_ratio",
             py::overload_cast< int >(&ParGDBBase::refRatio, py::const_),
             py::arg("level"))
    ;

#ifdef AMREX_PARTICLES
    // Concrete ParGDB owned by an AmrCore. Mirrors AmrCore::GetParGDB().
    py::class_< AmrParGDB, ParGDBBase >(m, "AmrParGDB")
        .def(py::init< AmrCore* >(), py::arg("amr_core"),
             // keep the AmrCore (arg 1 to the ctor) alive while the GDB lives
             py::keep_alive<1, 2>())
    ;
#endif
}
