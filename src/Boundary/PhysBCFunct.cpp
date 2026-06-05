/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Boundary/PhysBCFunct.H"

#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Vector.H>


void init_PhysBCFunct(py::module& m)
{
    using namespace amrex;

    py::class_<PhysBCFunctNoOp>(m, "PhysBCFunctNoOp",
            "A physical boundary condition functor that does nothing. "
            "Use this if there is nothing to fill outside the physical "
            "domain, e.g., for fully periodic domains.")
        .def(py::init<>())
        .def("__call__", &PhysBCFunctNoOp::operator(),
             py::arg("mf"), py::arg("dcomp"), py::arg("ncomp"),
             py::arg("nghost"), py::arg("time"), py::arg("bccomp"))
    ;

    py::class_<CpuBndryFuncFab>(m, "CpuBndryFuncFab",
            "A boundary function functor for cell-centered data on the "
            "host (CPU). Without a user function, it fills "
            "BCType.foextrap, BCType.hoextrap, BCType.reflect_even and "
            "BCType.reflect_odd boundaries; BCType.ext_dir (external "
            "Dirichlet) values are left untouched. Use PhysBCFunctUser "
            "to fill external Dirichlet boundaries from Python.")
        .def(py::init<>())
    ;

    py::class_<PhysBCFunct<CpuBndryFuncFab>>(m, "PhysBCFunct_CpuBndryFuncFab",
            "A physical boundary condition functor for cell-centered data, "
            "as amrex::PhysBCFunct<CpuBndryFuncFab> in C++. Fills domain "
            "boundary ghost cells based on boundary condition records.")
        .def(py::init<>())
        .def(py::init<Geometry const &, Vector<BCRec> const &,
                      CpuBndryFuncFab const &>(),
             py::arg("geom"), py::arg("bcr"), py::arg("f"))
        .def("define",
             py::overload_cast<Geometry const &, Vector<BCRec> const &,
                               CpuBndryFuncFab const &>(
                 &PhysBCFunct<CpuBndryFuncFab>::define),
             py::arg("geom"), py::arg("bcr"), py::arg("f"))
        .def("__call__", &PhysBCFunct<CpuBndryFuncFab>::operator(),
             py::arg("mf"), py::arg("icomp"), py::arg("ncomp"),
             py::arg("nghost"), py::arg("time"), py::arg("bccomp"))
    ;

    py::class_<pyAMReX::PhysBCFunctUser>(m, "PhysBCFunctUser",
            "A physical boundary condition functor calling back into "
            "Python. The user-provided callable receives "
            "(mf, dcomp, ncomp, nghost, time, bccomp) and is expected to "
            "fill the ghost cells of mf that lie outside the physical "
            "domain, e.g., for external Dirichlet (BCType.ext_dir) "
            "boundaries. The callback runs on the host.")
        .def(py::init<>())
        .def(py::init<pyAMReX::PhysBCFunctUser::UserFillFunc>(),
             py::arg("f"))
        .def("__call__", &pyAMReX::PhysBCFunctUser::operator(),
             py::arg("mf"), py::arg("dcomp"), py::arg("ncomp"),
             py::arg("nghost"), py::arg("time"), py::arg("bccomp"))
    ;
}
