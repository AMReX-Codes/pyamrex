/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_IntVect.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


void init_IntVect(py::module &m) {
    py::class_< IntVect >(m, "IntVect")
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto iv = obj.cast<IntVect>();
                 std::stringstream s;
                 s << iv;
                 return "<" + name + " " + s.str() + ">";
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
                 const size_t ii = (i > 0) ? i : AMREX_SPACEDIM - i;
                 if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(AMREX_SPACEDIM));
                 return v[ii];
             })
        .def("__setitem__",
             [](IntVect& v, const int i, const int& val) {
                 const size_t ii = (i > 0) ? i : AMREX_SPACEDIM - i;
                 if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(AMREX_SPACEDIM));
                 return v[ii] = val;
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
}
