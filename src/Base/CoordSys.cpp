/* Copyright 2021-2023 The AMReX Community
 *
 * Authors: Axel Huebl, Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_CoordSys.H>


void init_CoordSys(py::module& m)
{
    using namespace amrex;

    py::class_<CoordSys> coord_sys(m, "CoordSys");

    py::native_enum<CoordSys::CoordType>(coord_sys, "CoordType", "enum.IntEnum")
        .value("undef", CoordSys::CoordType::undef)
        .value("cartesian", CoordSys::CoordType::cartesian)
        .value("RZ", CoordSys::CoordType::RZ)
        .value("SPHERICAL", CoordSys::CoordType::SPHERICAL)
        .export_values()
        .finalize()
    ;

    coord_sys.def("__repr__",
             [](const CoordSys&) {
                 return "<amrex.CoordSys>";
             }
        )
        .def(py::init<>())
        .def(py::init<const CoordSys&>())

        .def("ok", &CoordSys::Ok)
        .def("Coord", &CoordSys::Coord)
        .def("SetCoord", &CoordSys::SetCoord)
        .def("CoordInt", &CoordSys::CoordInt)
        .def("IsSPHERICAL", &CoordSys::IsSPHERICAL)
        .def("IsRZ",&CoordSys::IsRZ )
        .def("IsCartesian", &CoordSys::IsCartesian)

        // ...
    ;

}
