/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_ParallelDescriptor.H>


void init_ParallelDescriptor(py::module &m)
{
    using namespace amrex;

    auto mpd = m.def_submodule("ParallelDescriptor");

    mpd.def("NProcs", py::overload_cast<>(&ParallelDescriptor::NProcs))
       .def("MyProc", py::overload_cast<>(&ParallelDescriptor::MyProc))
       .def("IOProcessor", py::overload_cast<>(&ParallelDescriptor::IOProcessor))
       .def("IOProcessorNumber", py::overload_cast<>(&ParallelDescriptor::IOProcessorNumber))
   ;
    // ...
}
