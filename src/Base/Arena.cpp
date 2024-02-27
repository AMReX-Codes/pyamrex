/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Arena.H>


void init_Arena(py::module &m) {
    using namespace amrex;

    py::class_< Arena >(m, "Arena")
        .def_static("initialize", &Arena::Initialize)
        .def_static("print_usage", &Arena::PrintUsage)
        .def_static("print_usage_to_files", &Arena::PrintUsageToFiles,
                    py::arg("filename"), py::arg("message"))
        .def_static("finalize", &Arena::Finalize)

        // these two can be true at the same time
        .def_property_readonly("is_device_accessible", &Arena::isDeviceAccessible)
        .def_property_readonly("is_host_accessible", &Arena::isHostAccessible)

        // the next three are mutually exclusive
        .def_property_readonly("is_managed", &Arena::isManaged)
        .def_property_readonly("is_device", &Arena::isDevice)
        .def_property_readonly("is_pinned", &Arena::isPinned)

#ifdef AMREX_USE_GPU
        .def_property_readonly("is_stream_ordered_arena", &Arena::isStreamOrderedArena)
#endif

        .def("has_free_device_memory", &Arena::hasFreeDeviceMemory,
             py::arg("sz"),
             "Does the device have enough free memory for allocating this "
             "much memory? For CPU builds, this always return true.")
    ;

    m.def("The_Arena", &The_Arena, py::return_value_policy::reference)
     .def("The_Async_Arena", &The_Async_Arena, py::return_value_policy::reference)
     .def("The_Device_Arena", &The_Device_Arena, py::return_value_policy::reference)
     .def("The_Managed_Arena", &The_Managed_Arena, py::return_value_policy::reference)
     .def("The_Pinned_Arena", &The_Pinned_Arena, py::return_value_policy::reference)
     .def("The_Cpu_Arena", &The_Cpu_Arena, py::return_value_policy::reference)
    ;

    // ArenaInfo
}
