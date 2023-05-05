/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_ParallelDescriptor.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <optional>

namespace py = nanobind;
using namespace amrex;


void init_ParallelDescriptor(py::module_ &m) {
    auto mpd = m.def_submodule("ParallelDescriptor");

    mpd.def("NProcs", py::overload_cast<>(&ParallelDescriptor::NProcs));
    // ...
}
