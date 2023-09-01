/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_Dim3.H>
#include <AMReX_IntVect.H>

#include <array>
#include <sstream>
#include <string>


void init_IntVect(py::module &m)
{
    using namespace amrex;

    py::class_< IntVect >(m, "IntVect")
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto iv = obj.cast<IntVect>();
                 std::stringstream s;
                 s << iv;
                 return "<amrex." + name + " " + s.str() + ">";
            }
        )
        .def("__str",
             [](const IntVect& iv) {
                 std::stringstream s;
                 s << iv;
                 return s.str();
             })

        .def(py::init<>())
#if (AMREX_SPACEDIM > 1)
        .def(py::init<AMREX_D_DECL(int, int, int)>())
#endif
        .def(py::init<int>())
        .def(py::init<const std::array<int, AMREX_SPACEDIM>&>())

        .def_property_readonly("sum", &IntVect::sum)
        .def_property_readonly("max",
            py::overload_cast<>(&IntVect::max, py::const_))
        .def_property_readonly("min",
            py::overload_cast<>(&IntVect::min, py::const_))
        .def_static("zero_vector", &IntVect::TheZeroVector)
        .def_static("unit_vector", &IntVect::TheUnitVector)
        .def_static("node_vector", &IntVect::TheNodeVector)
        .def_static("cell_vector", &IntVect::TheCellVector)
        .def_static("max_vector", &IntVect::TheMaxVector)
        .def_static("min_vector", &IntVect::TheMinVector)

        .def("dim3", &IntVect::dim3)
        .def("__getitem__",
             [](const IntVect& v, const int i) {
                 const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                 if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(AMREX_SPACEDIM));
                 return v[ii];
             })
        .def("__setitem__",
             [](IntVect& v, const int i, const int& val) {
                 const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                 if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(AMREX_SPACEDIM));
                 return v[ii] = val;
             })

        .def("__len__", [](IntVect const &) { return AMREX_SPACEDIM; })
        .def("__iter__", [](IntVect const & v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */

        .def("__eq__",
             py::overload_cast<int>(&IntVect::operator==, py::const_))
        .def("__eq__",
             py::overload_cast<const IntVect&>(&IntVect::operator==, py::const_))
        .def("__ne__",
             py::overload_cast<int>(&IntVect::operator!=, py::const_))
        .def("__ne__",
             py::overload_cast<const IntVect&>(&IntVect::operator!=, py::const_))
        .def("__lt__", &IntVect::operator<)
        .def("__le__", &IntVect::operator<=)
        .def("__gt__", &IntVect::operator>)
        .def("__ge__", &IntVect::operator>=)

        .def("__add__",
             py::overload_cast<int>(&IntVect::operator+, py::const_))
        .def("__add__",
             py::overload_cast<const IntVect&>(&IntVect::operator+, py::const_))
        .def("__sub__",
             py::overload_cast<int>(&IntVect::operator-, py::const_))
        .def("__sub__",
             py::overload_cast<const IntVect&>(&IntVect::operator-, py::const_))
        .def("__mul__",
             py::overload_cast<int>(&IntVect::operator*, py::const_))
        .def("__mul__",
             py::overload_cast<const IntVect&>(&IntVect::operator*, py::const_))
        .def("__truediv__",
             py::overload_cast<int>(&IntVect::operator/, py::const_))
        .def("__truediv__",
             py::overload_cast<const IntVect&>(&IntVect::operator/, py::const_))
        .def("__iadd__",
             py::overload_cast<int>(&IntVect::operator+=))
        .def("__iadd__",
             py::overload_cast<const IntVect&>(&IntVect::operator+=))
        .def("__isub__",
             py::overload_cast<int>(&IntVect::operator-=))
        .def("__isub__",
             py::overload_cast<const IntVect&>(&IntVect::operator-=))
        .def("__imul__",
             py::overload_cast<int>(&IntVect::operator*=))
        .def("__imul__",
             py::overload_cast<const IntVect&>(&IntVect::operator*=))
        .def("__itruediv__",
             py::overload_cast<int>(&IntVect::operator/=))
        .def("__itruediv__",
             py::overload_cast<const IntVect&>(&IntVect::operator/=))

        .def("numpy",
             [](const IntVect& iv) {
                 auto result = py::array(
                     py::buffer_info(
                         nullptr,
                         sizeof(int),
                         py::format_descriptor<int>::value,
                         1,
                         { AMREX_SPACEDIM },
                         { sizeof(int) }
                     ));
                 auto buf = result.request();
                 int* ptr = static_cast<int*>(buf.ptr);
                 for (int i=0; i < AMREX_SPACEDIM; ++i)
                     ptr[i] = iv[0];

                 return result;
             })
    ;

    m.def("coarsen",
         py::overload_cast<const IntVect&, const IntVect&>(&coarsen));
    m.def("coarsen",
          py::overload_cast<const Dim3&, const IntVect&>(&coarsen));
    m.def("coarsen",
          py::overload_cast<const IntVect&, int>(&coarsen));
    m.def("refine",
          py::overload_cast<const Dim3&, const IntVect&>(&refine));

    make_Vector<IntVect> (m, "IntVect");
}
