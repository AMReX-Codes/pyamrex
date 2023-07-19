#include "pyAMReX.H"

#include <AMReX_CoordSys.H>


void init_CoordSys(py::module& m)
{
    using namespace amrex;

    py::class_<CoordSys> coord_sys(m, "CoordSys");
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

    py::enum_<CoordSys::CoordType>(coord_sys, "CoordType")
        .value("undef", CoordSys::CoordType::undef)
        .value("cartesian", CoordSys::CoordType::cartesian)
        .value("RZ", CoordSys::CoordType::RZ)
        .value("SPHERICAL", CoordSys::CoordType::SPHERICAL)
        .export_values();
}
