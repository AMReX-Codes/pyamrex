/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: David Grote, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Dim3.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>

#include <array>
#include <sstream>
#include <string>


namespace {
    int check_index(const int i)
    {
        const int ii = (i >= 0) ? i : AMREX_SPACEDIM + i;
        if ((ii < 0) || (ii >= AMREX_SPACEDIM))
             throw py::index_error( "IndexType index " + std::to_string(i) + " out of bounds");
        return ii;
    }
}

void init_IndexType(py::module &m) {
    using namespace amrex;

    py::class_< IndexType > index_type(m, "IndexType");

    py::native_enum<IndexType::CellIndex>(index_type, "CellIndex", "enum.IntEnum")
        .value("CELL", IndexType::CellIndex::CELL)
        .value("NODE", IndexType::CellIndex::NODE)
        .export_values()
        .finalize()
    ;

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
                 const int ii = check_index(i);
                 return v[ii];
             })

        .def("__len__", [](IndexType const &) { return AMREX_SPACEDIM; })
        .def("__eq__",
             py::overload_cast<const IndexType&>(&IndexType::operator==, py::const_))
        .def("__ne__",
             py::overload_cast<const IndexType&>(&IndexType::operator!=, py::const_))
        .def("__lt__", &IndexType::operator<)

        .def("set", [](IndexType& v, int i) {
                 const int ii = check_index(i);
                 v.set(ii);
             })
        .def("unset", [](IndexType& v, int i) {
                 const int ii = check_index(i);
                 v.unset(ii);
             })
        .def("test", [](const IndexType& v, int i) {
                 const int ii = check_index(i);
                 return v.test(ii);
             })
        .def("setall", &IndexType::setall)
        .def("clear", &IndexType::clear)
        .def("any", &IndexType::any)
        .def("ok", &IndexType::ok)
        .def("flip", [](IndexType& v, int i) {
                 const int ii = check_index(i);
                 v.flip(ii);
             })

        .def("cell_centered", py::overload_cast<>(&IndexType::cellCentered, py::const_))
        .def("cell_centered", [](const IndexType& v, int i) {
                 const int ii = check_index(i);
                 return v.cellCentered(ii);
             })
        .def("node_centered", py::overload_cast<>(&IndexType::nodeCentered, py::const_))
        .def("node_centered", [](const IndexType& v, int i) {
                 const int ii = check_index(i);
                 return v.nodeCentered(ii);
             })

        .def("set_type", [](IndexType& v, int i, IndexType::CellIndex t) {
                 const int ii = check_index(i);
                 v.setType(ii, t);
             })
        .def("ix_type", py::overload_cast<>(&IndexType::ixType, py::const_))
        .def("ix_type", [](const IndexType& v, int i) {
                 const int ii = check_index(i);
                 return v.ixType(ii);
             })
        .def("to_IntVect", &IndexType::toIntVect)

        .def_static("cell_type", &IndexType::TheCellType)
        .def_static("node_type", &IndexType::TheNodeType)

        ;
}
