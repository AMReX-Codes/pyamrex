/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Arena.H>

#include <nanobind/nanobind.h>
#include <nanobind/numpy.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace amrex;


void init_Arena(py::module_ &m) {
    py::class_< Arena >(m, "Arena");

    m.def("The_Arena", &The_Arena, py::return_value_policy::reference)
     .def("The_Async_Arena", &The_Async_Arena, py::return_value_policy::reference)
     .def("The_Device_Arena", &The_Device_Arena, py::return_value_policy::reference)
     .def("The_Managed_Arena", &The_Managed_Arena, py::return_value_policy::reference)
     .def("The_Pinned_Arena", &The_Pinned_Arena, py::return_value_policy::reference)
     .def("The_Cpu_Arena", &The_Cpu_Arena, py::return_value_policy::reference)
    ;

    // ArenaInfo
}
