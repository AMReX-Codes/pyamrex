#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Dim3.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

void init_Dim3(py::module& m)
{
    py::class_<Dim3>(m, "Dim3")
        .def("__repr__",
             [](const Dim3& d) {
                 std::stringstream s;
                 s << d;
                 return "<pyamrex.Dim3 '" + s.str() + "'>";
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
        .def_readwrite("x", &Dim3::x)
        .def_readwrite("y", &Dim3::y)
        .def_readwrite("z", &Dim3::z)
        ;

    py::class_<XDim3>(m, "XDim3")
        .def(py::init<Real, Real, Real>())
        .def_readwrite("x", &XDim3::x)
        .def_readwrite("y", &XDim3::y)
        .def_readwrite("z", &XDim3::z)
        ;
}
