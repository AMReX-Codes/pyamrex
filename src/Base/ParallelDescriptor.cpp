/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_ParallelDescriptor.H>

#include <vector>


void init_ParallelDescriptor(py::module &m)
{
    using namespace amrex;

    auto mpd = m.def_submodule("ParallelDescriptor");

    mpd.def("NProcs", py::overload_cast<>(&ParallelDescriptor::NProcs))
       .def("MyProc", py::overload_cast<>(&ParallelDescriptor::MyProc))
       .def("IOProcessor", py::overload_cast<>(&ParallelDescriptor::IOProcessor))
       .def("IOProcessorNumber", py::overload_cast<>(&ParallelDescriptor::IOProcessorNumber))

       .def("Barrier", [](){ ParallelDescriptor::Barrier(); })

       // note: in Python, the reduced values are returned (arguments
       // are not modified in place)
       .def("ReduceRealMin",
            [](Real v){ ParallelDescriptor::ReduceRealMin(v); return v; })
       .def("ReduceRealMin",
            [](std::vector<Real> v){
                ParallelDescriptor::ReduceRealMin(v.data(), int(v.size()));
                return v;
            })
       .def("ReduceRealMax",
            [](Real v){ ParallelDescriptor::ReduceRealMax(v); return v; })
       .def("ReduceRealMax",
            [](std::vector<Real> v){
                ParallelDescriptor::ReduceRealMax(v.data(), int(v.size()));
                return v;
            })
       .def("ReduceRealSum",
            [](Real v){ ParallelDescriptor::ReduceRealSum(v); return v; })
       .def("ReduceRealSum",
            [](std::vector<Real> v){
                ParallelDescriptor::ReduceRealSum(v.data(), int(v.size()));
                return v;
            })

       .def("ReduceIntMin",
            [](int v){ ParallelDescriptor::ReduceIntMin(v); return v; })
       .def("ReduceIntMax",
            [](int v){ ParallelDescriptor::ReduceIntMax(v); return v; })
       .def("ReduceIntSum",
            [](int v){ ParallelDescriptor::ReduceIntSum(v); return v; })

       .def("ReduceLongMin",
            [](Long v){ ParallelDescriptor::ReduceLongMin(v); return v; })
       .def("ReduceLongMax",
            [](Long v){ ParallelDescriptor::ReduceLongMax(v); return v; })
       .def("ReduceLongSum",
            [](Long v){ ParallelDescriptor::ReduceLongSum(v); return v; })

       .def("ReduceBoolAnd",
            [](bool v){ ParallelDescriptor::ReduceBoolAnd(v); return v; })
       .def("ReduceBoolOr",
            [](bool v){ ParallelDescriptor::ReduceBoolOr(v); return v; })
   ;
}
