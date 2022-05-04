#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_CoordSys.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

void init_Geometry(py::module& m)
{
    py::class_<Geometry, CoordSys>(m, "Geometry")
        .def("__repr__",
             [](const Geometry&) {
                 return "<amrex.Geometry>";
             }
        )
        .def(py::init<>())
        .def(py::init<
            const Box&,
            const RealBox&,
            int,
            Array<int, AMREX_SPACEDIM> const&
          >(),
          py::arg("dom"), py::arg("rb"), py::arg("coord"), py::arg("is_per"))

          // ...
    ;

}
