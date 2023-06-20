/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: David Grote
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <array>
#include <sstream>
#include <string>

namespace py = pybind11;
using namespace amrex;


void init_IndexType(py::module &m) {
    py::class_< IndexType > index_type(m, "IndexType");
    index_type.def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto iv = obj.cast<IndexType>();
                 std::stringstream s;
                 s << iv;
                 return "<amrex." + name + " " + s.str() + ">";
            }
        )
        .def("__str",
             [](const IndexType& iv) {
                 std::stringstream s;
                 s << iv;
                 return s.str();
             })

        .def(py::init<>())
        .def(py::init<IndexType>())
#if (AMREX_SPACEDIM > 1)
        .def(py::init<AMREX_D_DECL(IndexType::CellIndex, IndexType::CellIndex, IndexType::CellIndex)>())
#endif

        .def("__getitem__",
             [](const IndexType& v, const int i) {
                 const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
                 if ((ii < 0) || (ii >= AMREX_SPACEDIM))
                     throw py::index_error(
                         "Index must be between 0 and " +
                         std::to_string(AMREX_SPACEDIM));
                 return v[ii];
             })

        .def("__len__", [](IndexType const &) { return AMREX_SPACEDIM; })
        .def("__eq__",
             py::overload_cast<const IndexType&>(&IndexType::operator==, py::const_))
        .def("__ne__",
             py::overload_cast<const IndexType&>(&IndexType::operator!=, py::const_))
        .def("__lt__", &IndexType::operator<)

        .def("set", &IndexType::set)
        .def("unset", &IndexType::unset)
        .def("test", &IndexType::test)
        .def("setall", &IndexType::setall)
        .def("clear", &IndexType::clear)
        .def("any", &IndexType::any)
        .def("ok", &IndexType::ok)
        .def("flip", &IndexType::flip)

        .def("cellCentered", py::overload_cast<>(&IndexType::cellCentered, py::const_))
        .def("cellCentered", py::overload_cast<int>(&IndexType::cellCentered, py::const_))
        .def("nodeCentered", py::overload_cast<>(&IndexType::nodeCentered, py::const_))
        .def("nodeCentered", py::overload_cast<int>(&IndexType::nodeCentered, py::const_))

        .def("set_type", &IndexType::setType)
        .def("ix_type", py::overload_cast<int>(&IndexType::ixType, py::const_))
        .def("ix_type", py::overload_cast<>(&IndexType::ixType, py::const_))
        .def("to_IntVect", &IndexType::toIntVect)

        .def_static("cell_type", &IndexType::TheCellType)
        .def_static("node_type", &IndexType::TheNodeType)

        ;

    py::enum_<IndexType::CellIndex>(index_type, "CellIndex")
        .value("CELL", IndexType::CellIndex::CELL)
        .value("NODE", IndexType::CellIndex::NODE)
        .export_values();

}
