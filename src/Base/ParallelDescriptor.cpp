/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_ParallelDescriptor.H>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <sstream>
#include <optional>

namespace py = pybind11;
using namespace amrex;


void init_ParallelDescriptor(py::module &m) {
    auto mpd = m.def_submodule("ParallelDescriptor");

    mpd.def("NProcs", py::overload_cast<>(&ParallelDescriptor::NProcs));
    // ...
}
