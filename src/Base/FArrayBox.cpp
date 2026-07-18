/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_FArrayBox.H>

#include <istream>
#include <ostream>
#include <string>


void init_FArrayBox(nb::module_ &m) {
    using namespace amrex;

    nb::class_< FArrayBox, BaseFab<Real> >(m, "FArrayBox")
        .def("__repr__",
             [](FArrayBox const & /* fab */) {
                 std::string r = "<amrex.FArrayBox>";
                 return r;
             }
        )

        .def(nb::init< >())
        .def(nb::init< Arena* >())
        .def(nb::init< Box const &, int, Arena* >())
        .def(nb::init< Box const &, int, bool, bool, Arena* >())
        //.def(nb::init< FArrayBox const &, MakeType, int, int >())
        .def(nb::init< Box const &, int, Real const* >())
        .def(nb::init< Box const &, int, Real* >())
        .def(nb::init< Array4<Real> const& >(), nb::keep_alive<1, 2>())
        .def(nb::init< Array4<Real> const&, IndexType >(), nb::keep_alive<1, 2>())
        .def(nb::init< Array4<Real const> const& >(), nb::keep_alive<1, 2>())
        .def(nb::init< Array4<Real const> const&, IndexType >(), nb::keep_alive<1, 2>())

        /*
        .def("read_from",
             nb::overload_cast<std::istream&>(&FArrayBox::readFrom),
             nb::arg("is")
        )
        .def("read_from",
             nb::overload_cast<std::istream&, int>(&FArrayBox::readFrom),
             nb::arg("is"), nb::arg("compIndex")
        )
        .def("write_on",
             nb::overload_cast<std::ostream&>(&FArrayBox::writeOn, nb::const_),
             nb::arg("of")
        )
        .def("write_on",
             nb::overload_cast<std::ostream&, int, int>(&FArrayBox::writeOn, nb::const_),
             nb::arg("of"), nb::arg("comp"), nb::arg("num_comp")
        )
        */
    ;
}
