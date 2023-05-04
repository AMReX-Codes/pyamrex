#include <AMReX_Config.H>
#include <AMReX_CoordSys.H>

#include <nanobind/nanobind.h>
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

namespace py = nanobind;
using namespace amrex;

void init_CoordSys(py::module_& m)
{
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
