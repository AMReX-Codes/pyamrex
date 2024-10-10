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


namespace
{
    template<int dim>
    void init_IntVectND(py::module &m)
    {
        using namespace amrex;

        auto const iv_name = std::string("IntVect").append(std::to_string(dim)).append("D");
        using iv_type = IntVectND<dim>;

        py::class_< iv_type > py_iv(m, iv_name.c_str());
        py_iv
            .def("__repr__",
                 [iv_name](const iv_type& iv) {
                     std::stringstream s;
                     s << iv;
                     return "<amrex." + iv_name + s.str() + ">";
                 }
            )
            .def("__str",
                 [](const iv_type& iv) {
                     std::stringstream s;
                     s << iv;
                     return s.str();
                 })
        ;

        if constexpr (dim == 2) {
            py_iv.def(py::init<int, int>());
        } else if constexpr (dim == 3) {
            py_iv.def(py::init<int, int, int>());
        }

        py_iv
            .def(py::init<>())
            .def(py::init<int>())
            .def(py::init<const std::array<int, dim>&>())

            .def_property_readonly("sum", &iv_type::sum)
            .def_property_readonly("max",
                                   py::overload_cast<>(&iv_type::max, py::const_))
            .def_property_readonly("min",
                                   py::overload_cast<>(&iv_type::min, py::const_))
            .def_static("zero_vector", &iv_type::TheZeroVector)
            .def_static("unit_vector", &iv_type::TheUnitVector)
            .def_static("node_vector", &iv_type::TheNodeVector)
            .def_static("cell_vector", &iv_type::TheCellVector)
            .def_static("max_vector", &iv_type::TheMaxVector)
            .def_static("min_vector", &iv_type::TheMinVector)
        ;

        if constexpr (dim >= 1 && dim <=3) {
            py_iv.def("dim3",
               py::overload_cast<>(&iv_type::template dim3<dim>, py::const_));
        }

        py_iv
            .def("__getitem__",
                 [](const iv_type& v, const int i) {
                     const int ii = (i >= 0) ? i : dim + i;
                     if ((ii < 0) || (ii >= dim))
                         throw py::index_error(
                                 "Index must be between 0 and " +
                                 std::to_string(dim));
                     return v[ii];
                 })
            .def("__setitem__",
                 [](iv_type& v, const int i, const int& val) {
                     const int ii = (i >= 0) ? i : dim + i;
                     if ((ii < 0) || (ii >= dim))
                         throw py::index_error(
                                 "Index must be between 0 and " +
                                 std::to_string(dim));
                     return v[ii] = val;
                 })

            .def("__len__", [](iv_type const &) { return dim; })
            .def("__iter__", [](iv_type const & v) {
                return py::make_iterator(v.begin(), v.end());
            }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */

            .def("__eq__",
                 py::overload_cast<int>(&iv_type::operator==, py::const_))
            .def("__eq__",
                 py::overload_cast<const iv_type&>(&iv_type::operator==, py::const_))
            .def("__ne__",
                 py::overload_cast<int>(&iv_type::operator!=, py::const_))
            .def("__ne__",
                 py::overload_cast<const iv_type&>(&iv_type::operator!=, py::const_))
            .def("__lt__", &iv_type::operator<)
            .def("__le__", &iv_type::operator<=)
            .def("__gt__", &iv_type::operator>)
            .def("__ge__", &iv_type::operator>=)

            .def("__add__",
                 py::overload_cast<int>(&iv_type::operator+, py::const_))
            .def("__add__",
                 py::overload_cast<const iv_type&>(&iv_type::operator+, py::const_))
            .def("__sub__",
                 py::overload_cast<int>(&iv_type::operator-, py::const_))
            .def("__sub__",
                 py::overload_cast<const iv_type&>(&iv_type::operator-, py::const_))
            .def("__mul__",
                 py::overload_cast<int>(&iv_type::operator*, py::const_))
            .def("__mul__",
                 py::overload_cast<const iv_type&>(&iv_type::operator*, py::const_))
            .def("__truediv__",
                 py::overload_cast<int>(&iv_type::operator/, py::const_))
            .def("__truediv__",
                 py::overload_cast<const iv_type&>(&iv_type::operator/, py::const_))
            .def("__iadd__",
                 py::overload_cast<int>(&iv_type::operator+=))
            .def("__iadd__",
                 py::overload_cast<const iv_type&>(&iv_type::operator+=))
            .def("__isub__",
                 py::overload_cast<int>(&iv_type::operator-=))
            .def("__isub__",
                 py::overload_cast<const iv_type&>(&iv_type::operator-=))
            .def("__imul__",
                 py::overload_cast<int>(&iv_type::operator*=))
            .def("__imul__",
                 py::overload_cast<const iv_type&>(&iv_type::operator*=))
            .def("__itruediv__",
                 py::overload_cast<int>(&iv_type::operator/=))
            .def("__itruediv__",
                 py::overload_cast<const iv_type&>(&iv_type::operator/=))

            .def("numpy",
                 [](const iv_type& iv) {
                     auto result = py::array(
                             py::buffer_info(
                                     nullptr,
                                     sizeof(int),
                                     py::format_descriptor<int>::value,
                                     1,
                                     { dim },
                                     { sizeof(int) }
                             ));
                     auto buf = result.request();
                     int* ptr = static_cast<int*>(buf.ptr);
                     for (int i=0; i < dim; ++i)
                         ptr[i] = iv[0];

                     return result;
                 })
        ;

        m.def("coarsen",
              py::overload_cast<const iv_type&, const iv_type&>(&coarsen<dim>));
        m.def("coarsen",
              py::overload_cast<const Dim3&, const iv_type&>(&coarsen<dim>));
        m.def("coarsen",
              py::overload_cast<const iv_type&, int>(&coarsen<dim>));
        m.def("refine",
              py::overload_cast<const Dim3&, const iv_type&>(&refine<dim>));
    }
}


void init_IntVect(py::module &m)
{
    using namespace amrex;

    init_IntVectND<1>(m);
    init_IntVectND<2>(m);
    init_IntVectND<3>(m);

    // alias for IntVect in current module's dim
    auto const iv_name = std::string("IntVect").append(std::to_string(AMREX_SPACEDIM)).append("D");
    m.attr("IntVect") = m.attr(iv_name.c_str());

    make_Vector<IntVect> (m, "IntVect");
}
