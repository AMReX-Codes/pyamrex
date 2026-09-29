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
#ifdef AMREX_USE_MPI
    mpd.def("Communicator",
        []() { return static_cast<int>(MPI_Comm_c2f(ParallelDescriptor::Communicator())); },
        R"pbdoc(
The MPI communicator of AMReX, as a Fortran handle (int).

Use ``mpi4py.MPI.Comm.f2py(amr.ParallelDescriptor.Communicator())`` to get an
mpi4py communicator. It is owned by AMReX: do not free it.
)pbdoc"
    );
#endif
    // ...
}
