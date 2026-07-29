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
    void init_IntVectND(nb::module_ &m)
    {
        using namespace amrex;

        auto const iv_name = std::string("IntVect").append(std::to_string(dim)).append("D");
        using iv_type = IntVectND<dim>;

        nb::class_< iv_type > py_iv(m, iv_name.c_str());
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
            py_iv.def(nb::init<int, int>());
        } else if constexpr (dim == 3) {
            py_iv.def(nb::init<int, int, int>());
        }

        py_iv
            .def(nb::init<>())
            .def(nb::init<int>())
            .def(nb::init<const std::array<int, dim>&>())

            .def_prop_ro("sum", &iv_type::sum)
            .def_prop_ro("max",
                                   nb::overload_cast<>(&iv_type::max, nb::const_))
            .def_prop_ro("min",
                                   nb::overload_cast<>(&iv_type::min, nb::const_))
            .def_static("zero_vector", &iv_type::TheZeroVector)
            .def_static("unit_vector", &iv_type::TheUnitVector)
            .def_static("node_vector", &iv_type::TheNodeVector)
            .def_static("cell_vector", &iv_type::TheCellVector)
            .def_static("max_vector", &iv_type::TheMaxVector)
            .def_static("min_vector", &iv_type::TheMinVector)
        ;

        if constexpr (dim >= 1 && dim <=3) {
            py_iv.def("dim3",
               [](const iv_type& iv) { return iv.dim3(); });
        }

        py_iv
            .def("__getitem__",
                 [](const iv_type& v, const int i) {
                     const int ii = (i >= 0) ? i : dim + i;
                     if ((ii < 0) || (ii >= dim)) {
                         auto message = "Index must be between 0 and " +
                                        std::to_string(dim);
                         throw nb::index_error(message.c_str());
                     }
                     return v[ii];
                 })
            .def("__setitem__",
                 [](iv_type& v, const int i, const int& val) {
                     const int ii = (i >= 0) ? i : dim + i;
                     if ((ii < 0) || (ii >= dim)) {
                         auto message = "Index must be between 0 and " +
                                        std::to_string(dim);
                         throw nb::index_error(message.c_str());
                     }
                     return v[ii] = val;
                 })

            .def("__len__", [](iv_type const &) { return dim; })
            .def("__iter__", [](iv_type const & v) {
                return nb::make_iterator(
                    nb::type<iv_type>(), "Iterator", v.begin(), v.end()
                );
            }, nb::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */

            .def("__eq__",
                 nb::overload_cast<int>(&iv_type::operator==, nb::const_))
            .def("__eq__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator==, nb::const_))
            .def("__ne__",
                 nb::overload_cast<int>(&iv_type::operator!=, nb::const_))
            .def("__ne__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator!=, nb::const_))
            .def("__lt__", &iv_type::operator<)
            .def("__le__", &iv_type::operator<=)
            .def("__gt__", &iv_type::operator>)
            .def("__ge__", &iv_type::operator>=)

            .def("__add__",
                 nb::overload_cast<int>(&iv_type::operator+, nb::const_))
            .def("__add__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator+, nb::const_))
            .def("__sub__",
                 nb::overload_cast<int>(&iv_type::operator-, nb::const_))
            .def("__sub__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator-, nb::const_))
            .def("__mul__",
                 nb::overload_cast<int>(&iv_type::operator*, nb::const_))
            .def("__mul__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator*, nb::const_))
            .def("__truediv__",
                 nb::overload_cast<int>(&iv_type::operator/, nb::const_))
            .def("__truediv__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator/, nb::const_))
            .def("__iadd__",
                 nb::overload_cast<int>(&iv_type::operator+=))
            .def("__iadd__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator+=))
            .def("__isub__",
                 nb::overload_cast<int>(&iv_type::operator-=))
            .def("__isub__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator-=))
            .def("__imul__",
                 nb::overload_cast<int>(&iv_type::operator*=))
            .def("__imul__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator*=))
            .def("__itruediv__",
                 nb::overload_cast<int>(&iv_type::operator/=))
            .def("__itruediv__",
                 nb::overload_cast<const iv_type&>(&iv_type::operator/=))

            .def("numpy",
                 [](const iv_type& iv) {
                     auto numpy = nb::module_::import_("numpy");
                     auto result = numpy.attr("empty")(
                         nb::make_tuple(dim),
                         numpy.attr("dtype")(pyAMReX::buffer_format<int>())
                     );
                     auto array = nb::cast<nb::ndarray<nb::numpy, int, nb::ndim<1>>>(result);
                     int* ptr = array.data();
                     for (int i=0; i < dim; ++i)
                         ptr[i] = iv[0];

                     return result;
                 })
        ;

        m.def("coarsen",
              nb::overload_cast<const iv_type&, const iv_type&>(&coarsen<dim>));
        m.def("coarsen",
              nb::overload_cast<const Dim3&, const iv_type&>(&coarsen<dim>));
        m.def("coarsen",
              nb::overload_cast<const iv_type&, int>(&coarsen<dim>));
        m.def("refine",
              nb::overload_cast<const Dim3&, const iv_type&>(&refine<dim>));
    }
}


void init_IntVect(nb::module_ &m)
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
