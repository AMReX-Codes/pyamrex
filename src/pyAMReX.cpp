/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX.H>

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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = nanobind;


// forward declarations of exposed classes
void init_AMReX(py::module_&);
void init_Arena(py::module_&);
void init_Array4(py::module_&);
void init_BaseFab(py::module_&);
void init_Box(py::module_ &);
void init_RealBox(py::module_ &);
void init_BoxArray(py::module_ &);
void init_CoordSys(py::module_&);
void init_Dim3(py::module_&);
void init_DistributionMapping(py::module_&);
void init_FArrayBox(py::module_&);
void init_Geometry(py::module_&);
void init_IntVect(py::module_ &);
void init_RealVect(py::module_ &);
void init_AmrMesh(py::module_ &);
void init_MultiFab(py::module_ &);
void init_ParallelDescriptor(py::module_ &);
void init_ParmParse(py::module_ &);
void init_Particle(py::module_ &);
void init_StructOfArrays(py::module_ &);
void init_ArrayOfStructs(py::module_ &);
void init_ParticleTile(py::module_ &);
void init_ParticleContainer(py::module_ &);
void init_Periodicity(py::module_ &);
void init_PODVector(py::module_ &);
void init_Vector(py::module_ &);



NB_MODULE(amrex_pybind, m) {
    m.doc() = R"pbdoc(
            amrex_pybind
            -----------
            .. currentmodule:: amrex_pybind

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
               RealVect
               MultiFab
               ParallelDescriptor
               Particle
               ParmParse
               ParticleTile
               ParticleContainer
               Periodicity
               PODVector
               StructOfArrays
               Vector
    )pbdoc";

    // note: order from parent to child classes
    init_AMReX(m);
    init_Arena(m);
    init_Dim3(m);
    init_IntVect(m);
    init_RealVect(m);
    init_Periodicity(m);
    init_Array4(m);
    init_Box(m);
    init_BoxArray(m);
    init_ParmParse(m);
    init_BaseFab(m);
    init_FArrayBox(m);
    init_MultiFab(m);
    init_DistributionMapping(m);
    init_RealBox(m);
    init_CoordSys(m);
    init_Geometry(m);
    init_ParallelDescriptor(m);
    init_Particle(m);
    init_StructOfArrays(m);
    init_ArrayOfStructs(m);
    init_ParticleTile(m);
    init_PODVector(m);
    init_Vector(m);
    init_ParticleContainer(m);
    init_AmrMesh(m);

    // API runtime version
    //   note PEP-440 syntax: x.y.zaN but x.y.z.devN
#ifdef PYAMReX_VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(PYAMReX_VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // authors
    m.attr("__author__") =
        "Axel Huebl, Ryan Sandberg, Shreyas Ananthan, Remi Lehe, "
        "Weiqun Zhang, et al.";

    // API runtime build-time feature variants
    // m.attr("variants") = amrex::getVariants();
    // TODO allow to query runtime versions of all dependencies

    // license SPDX identifier
    m.attr("__license__") = "BSD-3-Clause-LBNL";

    // TODO broken numpy if not at least v1.15.0: raise warning
    // auto numpy = py::module_::import("numpy");
    // auto npversion = numpy.attr("__version__");
    // std::cout << "numpy version: " << py::str(npversion) << std::endl;
}
