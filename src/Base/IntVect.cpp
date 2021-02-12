/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_IntVect.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


void init_IntVect(py::module &m) {
    py::class_< IntVect >(m, "Int_Vect")
        .def("__repr__",
            [](IntVect const & iv) {
                std::stringstream s;
                s << iv;
                return "<amrex.Int_Vect '" + s.str() + "'>";
            }
        )

        .def_property_readonly("sum", &IntVect::sum)
        .def_property_readonly("max",
            py::overload_cast<>(&IntVect::max, py::const_))
        .def_property_readonly("min",
            py::overload_cast<>(&IntVect::min, py::const_))
        .def_property_readonly("the_zero_vector", &IntVect::TheZeroVector)
        .def_property_readonly("the_unit_vector", &IntVect::TheUnitVector)
        .def_property_readonly("the_node_vector", &IntVect::TheNodeVector)
        .def_property_readonly("the_cell_vector", &IntVect::TheCellVector)
        .def_property_readonly("the_max_vector", &IntVect::TheMaxVector)
        .def_property_readonly("the_min_vector", &IntVect::TheMinVector)

        //.def("to_array", &IntVect::toArray)
        // maxDir

        // __getitem__
        // __iter__

    ;
}
