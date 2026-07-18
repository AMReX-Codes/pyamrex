#include "pyAMReX.H"

#include <AMReX_Dim3.H>

#include <sstream>


void init_Dim3(nb::module_& m)
{
    using namespace amrex;

    nb::class_<Dim3>(m, "Dim3")
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
        .def(nb::init<int, int, int>())
        .def_rw("x", &Dim3::x)
        .def_rw("y", &Dim3::y)
        .def_rw("z", &Dim3::z)
        ;

    nb::class_<XDim3>(m, "XDim3")
        .def(nb::init<Real, Real, Real>())
        .def_rw("x", &XDim3::x)
        .def_rw("y", &XDim3::y)
        .def_rw("z", &XDim3::z)
        ;
}
