/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_ParallelDescriptor.H>


void init_ParallelDescriptor(nb::module_ &m)
{
    using namespace amrex;

    auto mpd = m.def_submodule("ParallelDescriptor");

    mpd.def("NProcs", nb::overload_cast<>(&ParallelDescriptor::NProcs))
       .def("MyProc", nb::overload_cast<>(&ParallelDescriptor::MyProc))
       .def("IOProcessor", nb::overload_cast<>(&ParallelDescriptor::IOProcessor))
       .def("IOProcessorNumber", nb::overload_cast<>(&ParallelDescriptor::IOProcessorNumber))
   ;
    // ...
}
