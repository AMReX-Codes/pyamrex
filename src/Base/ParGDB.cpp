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


void init_ParGDB (nb::module_& m)
{
    using namespace amrex;

    // Abstract base: the Geometry/DistributionMapping/BoxArray broker handed to
    // a ParticleContainer. No Python constructor (it is pure virtual); concrete
    // instances come from AmrParGDB or are built C++-side.
    nb::class_< ParGDBBase >(
        m, "ParGDBBase",
        R"pbdoc(
Abstract broker for particle geometry, box arrays and distribution maps.

Particle containers use a ``ParGDBBase`` to query mesh metadata for each AMR
level.  Python users usually obtain a concrete ``AmrParGDB`` from
``AmrCore.get_par_gdb()``.
)pbdoc")
        .def("particle_geom",
             nb::overload_cast< int >(&ParGDBBase::ParticleGeom, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return particle Geometry for AMR level.")
        .def("geom",
             nb::overload_cast< int >(&ParGDBBase::Geom, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return mesh Geometry for AMR level.")
        .def("particle_dist_map",
             nb::overload_cast< int >(&ParGDBBase::ParticleDistributionMap, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return particle DistributionMapping for AMR level.")
        .def("dist_map",
             nb::overload_cast< int >(&ParGDBBase::DistributionMap, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return mesh DistributionMapping for AMR level.")
        .def("particle_box_array",
             nb::overload_cast< int >(&ParGDBBase::ParticleBoxArray, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return particle BoxArray for AMR level.")
        .def("box_array",
             nb::overload_cast< int >(&ParGDBBase::boxArray, nb::const_),
             nb::rv_policy::reference_internal, nb::arg("level"),
             "Return mesh BoxArray for AMR level.")

        .def("set_particle_box_array", &ParGDBBase::SetParticleBoxArray,
             nb::arg("level"), nb::arg("new_ba"),
             "Replace the particle BoxArray for AMR level.")
        .def("set_particle_dist_map", &ParGDBBase::SetParticleDistributionMap,
             nb::arg("level"), nb::arg("new_dm"),
             "Replace the particle DistributionMapping for AMR level.")
        .def("set_particle_geometry", &ParGDBBase::SetParticleGeometry,
             nb::arg("level"), nb::arg("new_geom"),
             "Replace the particle Geometry for AMR level.")

        .def("level_defined", &ParGDBBase::LevelDefined, nb::arg("level"),
             "Return True if AMR level has valid mesh metadata.")
        .def("finest_level", &ParGDBBase::finestLevel,
             "Return the finest currently defined AMR level.")
        .def("max_level", &ParGDBBase::maxLevel,
             "Return the maximum AMR level supported by this broker.")
        .def("ref_ratio",
             nb::overload_cast< int >(&ParGDBBase::refRatio, nb::const_),
             nb::arg("level"),
             "Return the refinement ratio from level to level + 1.")
    ;

#ifdef AMREX_PARTICLES
    // Concrete ParGDB owned by an AmrCore. Mirrors AmrCore::GetParGDB().
    nb::class_< AmrParGDB, ParGDBBase >(
        m, "AmrParGDB",
        "Concrete particle metadata broker backed by an AmrCore.")
        .def(nb::init< AmrCore* >(), nb::arg("amr_core"),
             // keep the AmrCore (arg 1 to the ctor) alive while the GDB lives
             nb::keep_alive<1, 2>(),
             "Construct a particle metadata broker backed by amr_core.")
    ;
#endif
}
