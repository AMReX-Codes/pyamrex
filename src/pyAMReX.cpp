/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX.H>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


// forward declarations of exposed classes
void init_Algorithm(py::module&);
void init_AMReX(py::module&);
void init_Arena(py::module&);
void init_Array4(py::module&);
void init_BaseFab(py::module&);
void init_Box(py::module &);
void init_RealBox(py::module &);
void init_BoxArray(py::module &);
void init_CoordSys(py::module&);
void init_Dim3(py::module&);
void init_DistributionMapping(py::module&);
void init_FArrayBox(py::module&);
void init_Geometry(py::module&);
void init_IndexType(py::module &);
void init_IntVect(py::module &);
void init_RealVect(py::module &);
void init_AmrMesh(py::module &);
void init_MultiFab(py::module &);
void init_ParallelDescriptor(py::module &);
void init_ParmParse(py::module &);
void init_ParticleContainer(py::module &);
void init_Periodicity(py::module &);
void init_PlotFileUtil(py::module &);
void init_PODVector(py::module &);
void init_Utility(py::module &);
void init_Vector(py::module &);
void init_Version(py::module &);
void init_VisMF(py::module &);
#ifdef AMREX_USE_MPI
void init_MPMD(py::module &);
#endif

#if AMREX_SPACEDIM == 1
PYBIND11_MODULE(amrex_1d_pybind, m) {
#elif AMREX_SPACEDIM == 2
PYBIND11_MODULE(amrex_2d_pybind, m) {
#elif AMREX_SPACEDIM == 3
PYBIND11_MODULE(amrex_3d_pybind, m) {
#else
#  error "AMREX_SPACEDIM must be 1, 2 or 3"
#endif
    m.doc() = R"pbdoc(
            amrex
            -----
            .. currentmodule:: amrex

            .. autosummary::
               :toctree: _generate
               AmrInfo
               AmrMesh
               Arena
               ArrayOfStructs
               Box
               RealBox
               BoxArray
               Dim3
               FArrayBox
               IntVect
               IndexType
               RealVect
               MultiFab
               ParallelDescriptor
               Particle
               ParmParse
               ParticleTile
               ParticleContainer
               Periodicity
               PlotFileUtil
               PODVector
               StructOfArrays
               Utility
               Vector
               VisMF
    )pbdoc";

    // note: order from parent to child classes and argument usage
    init_AMReX(m);
    init_Arena(m);
    init_Dim3(m);
    init_Algorithm(m);
    init_IntVect(m);
    init_IndexType(m);
    init_RealVect(m);
    init_Box(m);
    init_Periodicity(m);
    init_Array4(m);
    init_BoxArray(m);
    init_ParmParse(m);
    init_CoordSys(m);
    init_RealBox(m);
    init_Vector(m);
    init_Geometry(m);
    init_DistributionMapping(m);
    init_BaseFab(m);
    init_FArrayBox(m);
    init_MultiFab(m);
    init_ParallelDescriptor(m);
    init_PODVector(m);

    init_ParticleContainer(m);
    init_AmrMesh(m);

#ifdef AMREX_USE_MPI
    init_MPMD(m);
#endif
    // Wrappers around standalone functions
    init_PlotFileUtil(m);
    init_Utility(m);
    init_Version(m);
    init_VisMF(m);

    // authors
    m.attr("__author__") =
        "Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, "
        "Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang";

    // API runtime build-time feature variants
    // m.attr("variants") = amrex::getVariants();
    // TODO allow to query runtime versions of all dependencies

    // license SPDX identifier
    m.attr("__license__") = "BSD-3-Clause-LBNL";

    // TODO broken numpy if not at least v1.15.0: raise warning
    // auto numpy = py::module::import("numpy");
    // auto npversion = numpy.attr("__version__");
    // std::cout << "numpy version: " << py::str(npversion) << std::endl;
}
