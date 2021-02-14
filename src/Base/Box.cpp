/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


void init_Box(py::module &m) {
    py::class_< Box >(m, "Box")
        .def("__repr__",
            [](Box const & b) {
                std::stringstream s;
                s << b.size();
                return "<amrex.Box of size '" + s.str() + "'>";
            }
        )

        .def(py::init< IntVect const &, IntVect const & >())
        .def(py::init< IntVect const &, IntVect const &, IntVect const & >())
        //.def(py::init< IntVect const &, IntVect const &, IndexType >())

        /*
        .def_property("small_end",
            &Box::smallEnd,
            py::overload_cast< IntVect const & >(&Box::setSmall))
        .def_property("big_end",
            &Box::bigEnd,
            &Box::setBig)
        */
        .def_property("type",
            py::overload_cast<>(&Box::type, py::const_),
            &Box::setType)

        .def_property_readonly("ix_type", &Box::ixType)
        .def_property_readonly("size", &Box::size)
        .def_property_readonly("length",
            py::overload_cast<>(&Box::length, py::const_))
        .def_property_readonly("is_empty", &Box::isEmpty)
        .def_property_readonly("ok", &Box::ok)
        .def_property_readonly("cell_centered", &Box::cellCentered)
        .def_property_readonly("num_pts", &Box::numPts)
        .def_property_readonly("d_num_pts", &Box::d_numPts)
        .def_property_readonly("volume", &Box::volume)
        .def_property_readonly("the_unit_box", &Box::TheUnitBox)
        .def_property_readonly("is_square", &Box::isSquare)

        // loVect3d
        // hiVect3d
        // loVect
        // hiVect

        .def("contains",
            py::overload_cast< IntVect const & >(&Box::contains, py::const_))
        .def("strictly_contains",
            py::overload_cast< IntVect const & >(&Box::strictly_contains, py::const_))
        //.def("intersects", &Box::intersects)
        //.def("same_size", &Box::sameSize)
        //.def("same_type", &Box::sameType)
        //.def("normalize", &Box::normalize)
        // longside
        // shortside
        // index
        // atOffset
        // atOffset3d
        // setRange
        // shift
        // shiftHalf
        // convert
        // surroundingNodes
        // enclosedCells
        // minBox
        // chop
        // grow
        // growLo
        // growHi
        // refine
        // coarsen
        // next
        // coarsenable

        // __getitem__

    ;
}
