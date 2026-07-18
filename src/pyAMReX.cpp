/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX.H>
#include <AMReX_MFIter.H>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


// forward declarations of exposed classes
void init_Algorithm(nb::module_&);
void init_AMReX(nb::module_&);
void init_AmrCore(nb::module_ &);
void init_AmrCore_class(nb::module_ &);
void init_AmrMesh(nb::module_ &);
void init_Arena(nb::module_&);
void init_Array4(nb::module_&);
void init_BaseFab(nb::module_&);
void init_BCRec(nb::module_&);
void init_BCUtil(nb::module_&);
void init_Box(nb::module_ &);
void init_RealBox(nb::module_ &);
void init_BoxArray(nb::module_ &);
void init_CoordSys(nb::module_&);
void init_Dim3(nb::module_&);
void init_DistributionMapping(nb::module_&);
void init_FabArray(nb::module_ &);
void init_FArrayBox(nb::module_&);
void init_Geometry(nb::module_&);
void init_iMultiFab(nb::module_&);
void init_IndexType(nb::module_ &);
void init_IntVect(nb::module_ &);
void init_MFInfo(nb::module_ &);
#ifdef AMREX_USE_MPI
void init_MPMD(nb::module_ &);
#endif
void init_MultiFab(nb::module_ &, nb::class_< amrex::MFIter >&);
void init_ParallelDescriptor(nb::module_ &);
void init_ParGDB(nb::module_ &);
void init_ParmParse(nb::module_ &);
void init_ParticleContainer(nb::module_ &);
void init_Periodicity(nb::module_ &);
void init_PhysBCFunct(nb::module_ &);
void init_PlotFileUtil(nb::module_ &);
void init_PODVector(nb::module_ &);
void init_RealVect(nb::module_ &);
void init_SmallMatrix(nb::module_ &);
void init_TagBox(nb::module_ &);
void init_Utility(nb::module_ &);
void init_Vector(nb::module_ &);
void init_Version(nb::module_ &);
void init_VisMF(nb::module_ &);
#ifdef AMREX_USE_EB
void init_EB(nb::module_ &);
#endif

#if AMREX_SPACEDIM == 1
NB_MODULE(amrex_1d_pybind, m) {
#elif AMREX_SPACEDIM == 2
NB_MODULE(amrex_2d_pybind, m) {
#elif AMREX_SPACEDIM == 3
NB_MODULE(amrex_3d_pybind, m) {
#else
#  error "AMREX_SPACEDIM must be 1, 2 or 3"
#endif
    m.doc() = R"pbdoc(
            amrex
            -----
            .. currentmodule:: amrex

            .. autosummary::
               :toctree: _generate
               AmrCore
               AmrInfo
               AmrMesh
               AmrParGDB
               Arena
               ArrayOfStructs
               Box
               RealBox
               BoxArray
               BCRec
               BCType
               CpuBndryFuncFab
               Dim3
               FArrayBox
               iMultiFab
               IntVect
               IndexType
               RealVect
               MFInfo
               MFItInfo
               MultiFab
               ParallelDescriptor
               ParGDBBase
               Particle
               ParmParse
               ParticleTile
               ParticleContainer
               Periodicity
               PhysBCFunctNoOp
               PhysBCFunct_CpuBndryFuncFab
               PhysBCFunctUser
               PhysBCType
               PlotFileUtil
               PODVector
               SmallMatrix
               StructOfArrays
               TagBox
               TagBoxArray
               Utility
               Vector
               Vector_BCRec
               fill_domain_boundary
               setBC
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
    init_SmallMatrix(m);
    init_Vector(m);
    init_BoxArray(m);
    init_ParmParse(m);
    init_CoordSys(m);
    init_RealBox(m);
    init_Geometry(m);
    init_DistributionMapping(m);
    init_BaseFab(m);
    init_BCRec(m);  // after Box and Vector
    init_FArrayBox(m);
    nb::class_< amrex::MFIter > py_MFIter(m, "MFIter", nb::dynamic_attr());
    init_FabArray(m);
    init_MFInfo(m);
    init_iMultiFab(m);
    init_MultiFab(m, py_MFIter);
    init_BCUtil(m);       // after MultiFab, Geometry and BCRec
    init_PhysBCFunct(m);  // after MultiFab, Geometry and BCRec
    init_ParallelDescriptor(m);
    init_PODVector(m);

    // note: the AmrCore class is declared before ParGDB and the particle
    // containers (they reference it in member signatures), while its member
    // functions are added after ParGDB (get_par_gdb returns an AmrParGDB)
    init_TagBox(m);
    init_AmrMesh(m);
    init_AmrCore_class(m);      // after AmrMesh (its pybind base)
    init_ParGDB(m);             // after the AmrCore class declaration
    init_ParticleContainer(m);  // after ParGDB (constructible from it)
    init_AmrCore(m);            // after ParGDB (AmrParGDB in signatures)

#ifdef AMREX_USE_MPI
    init_MPMD(m);
#endif

    // Wrappers around standalone functions
    init_PlotFileUtil(m);
    init_Utility(m);
    init_Version(m);
    init_VisMF(m);

#ifdef AMREX_USE_EB
    init_EB(m);
#endif

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
    // auto numpy = nb::module_::import_("numpy");
    // auto npversion = numpy.attr("__version__");
    // std::cout << "numpy version: " << nb::str(npversion) << std::endl;
}
