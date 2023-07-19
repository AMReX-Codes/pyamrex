/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Arena.H>


void init_Arena(py::module &m) {
    using namespace amrex;

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
