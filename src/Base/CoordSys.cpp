#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_CoordSys.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

void init_CoordSys(py::module& m)
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

        // ...
    ;

    py::enum_<CoordSys::CoordType>(coord_sys, "CoordType")
        .value("undef", CoordSys::CoordType::undef)
        .value("cartesian", CoordSys::CoordType::cartesian)
        .value("RZ", CoordSys::CoordType::RZ)
        .value("SPHERICAL", CoordSys::CoordType::SPHERICAL)
        .export_values();
}
