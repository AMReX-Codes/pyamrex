/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_FArrayBox.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <istream>
#include <ostream>
#include <string>

namespace py = nanobind;
using namespace amrex;


void init_FArrayBox(py::module_ &m) {
    py::class_< FArrayBox, BaseFab<Real> >(m, "FArrayBox")
        .def("__repr__",
             [](FArrayBox const & /* fab */) {
                 std::string r = "<amrex.FArrayBox>";
                 return r;
             }
        )

        .def(py::init< >())
        .def(py::init< Arena* >())
        .def(py::init< Box const &, int, Arena* >())
        .def(py::init< Box const &, int, bool, bool, Arena* >())
        .def(py::init< FArrayBox const &, MakeType, int, int >())
        .def(py::init< Box const &, int, Real const* >())
        .def(py::init< Box const &, int, Real* >())
        .def(py::init< Array4<Real> const& >())
        .def(py::init< Array4<Real> const&, IndexType >())
        .def(py::init< Array4<Real const> const& >())
        .def(py::init< Array4<Real const> const&, IndexType >())

        .def("read_from",
             py::overload_cast<std::istream&>(&FArrayBox::readFrom),
             py::arg("is")
        )
        .def("read_from",
             py::overload_cast<std::istream&, int>(&FArrayBox::readFrom),
             py::arg("is"), py::arg("compIndex")
        )
        .def("write_on",
             py::overload_cast<std::ostream&>(&FArrayBox::writeOn, py::const_),
             py::arg("of")
        )
        .def("write_on",
             py::overload_cast<std::ostream&, int, int>(&FArrayBox::writeOn, py::const_),
             py::arg("of"), py::arg("comp"), py::arg("num_comp")
        )
    ;
}
