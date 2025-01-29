/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"
#include "MultiFab.H"

#include <AMReX_FabArray.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>


void init_iMultiFab(py::module &m)
{
    using namespace amrex;

    py::class_< iMultiFab, FabArray<IArrayBox> > py_iMultiFab(m, "iMultiFab");
    make_MultiFab(py_iMultiFab, "iMultiFab");

    m.def("copy_mfab", py::overload_cast< iMultiFab &, iMultiFab const &, int, int, int, int >(&iMultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"))
     .def("copy_mfab", py::overload_cast< iMultiFab &, iMultiFab const &, int, int, int, IntVect const & >(&iMultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"));
}
