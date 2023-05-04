#include <AMReX_Config.H>
#include <AMReX_Dim3.H>

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

void init_Dim3(py::module_& m)
{
    py::class_<Dim3>(m, "Dim3")
        .def("__repr__",
             [](const Dim3& d) {
                 std::stringstream s;
                 s << d;
                 return "<amrex.Dim3 '" + s.str() + "'>";
             }
        )
        .def("__str__",
             [](const Dim3& d) {
                 std::stringstream s;
                 s << d;
                 return s.str();
             }
        )
        .def(py::init<int, int, int>())
        .def_rw("x", &Dim3::x)
        .def_rw("y", &Dim3::y)
        .def_rw("z", &Dim3::z)
        ;

    py::class_<XDim3>(m, "XDim3")
        .def(py::init<Real, Real, Real>())
        .def_rw("x", &XDim3::x)
        .def_rw("y", &XDim3::y)
        .def_rw("z", &XDim3::z)
        ;
}
