/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"
#include "MultiFab.H"

#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <string>


void init_MultiFab(nb::module_ &m, nb::class_< amrex::MFIter > & py_MFIter)
{
    using namespace amrex;

    nb::class_< MultiFab, FabArray<FArrayBox> > py_MultiFab(m, "MultiFab", nb::dynamic_attr());

    py_MFIter
        .def("__repr__",
             [](MFIter const & mfi) {
                 std::string r = "<amrex.MFIter (";
                 if( !mfi.isValid() ) { r.append("in"); }
                 r.append("valid)>");
                 return r;
             }
        )
        .def(nb::init< FabArrayBase const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the FabArrayBase (argument 2) alive
             nb::keep_alive<1, 2>()
        )
        .def(nb::init< FabArrayBase const &, MFItInfo const & >(),
            nb::keep_alive<1, 2>()
        )

        .def(nb::init< MultiFab const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the MultiFab (argument 2) alive
            nb::keep_alive<1, 2>()
        )
        .def(nb::init< MultiFab const &, MFItInfo const & >(),
            nb::keep_alive<1, 2>()
        )

        .def(nb::init< iMultiFab const & >(),
            nb::keep_alive<1, 2>()
        )
        .def(nb::init< iMultiFab const &, MFItInfo const & >(),
            nb::keep_alive<1, 2>()
        )

        // helpers for iteration __next__
        .def("_incr", &MFIter::operator++)
        .def("finalize", &MFIter::Finalize)

        .def("tilebox", nb::overload_cast< >(&MFIter::tilebox, nb::const_))
        .def("tilebox", nb::overload_cast< IntVect const & >(&MFIter::tilebox, nb::const_))
        .def("tilebox", nb::overload_cast< IntVect const &, IntVect const & >(&MFIter::tilebox, nb::const_))

        .def("validbox", &MFIter::validbox)
        .def("fabbox", &MFIter::fabbox)

        .def("nodaltilebox",
            nb::overload_cast< int >(&MFIter::nodaltilebox, nb::const_),
            nb::arg("dir") = -1)

        .def("growntilebox",
            nb::overload_cast< const IntVect& >(&MFIter::growntilebox, nb::const_),
            nb::arg("ng") = -1000000)

        .def("grownnodaltilebox",
            nb::overload_cast< int, int >(&MFIter::grownnodaltilebox, nb::const_),
            nb::arg("int") = -1, nb::arg("ng") = -1000000)
        .def("grownnodaltilebox",
            nb::overload_cast< int, const IntVect& >(&MFIter::grownnodaltilebox, nb::const_),
            nb::arg("int"), nb::arg("ng"))

        .def_prop_ro("is_valid", &MFIter::isValid)
        .def_prop_ro("index", &MFIter::index)
        .def_prop_ro("length", &MFIter::length)
    ;

    m.def("htod_memcpy",
          nb::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&htod_memcpy<FArrayBox>),
          nb::arg("dest"), nb::arg("src"),
          "Copy from a host to device FabArray."
    );
    m.def("htod_memcpy",
          nb::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&htod_memcpy<FArrayBox>),
          nb::arg("dest"), nb::arg("src"), nb::arg("scomp"), nb::arg("dcomp"), nb::arg("ncomp"),
          "Copy from a host to device FabArray for a specific (number of) component(s)."
    );

    m.def("dtoh_memcpy",
          nb::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&dtoh_memcpy<FArrayBox>),
          nb::arg("dest"), nb::arg("src"),
          "Copy from a device to host FabArray."
    );
    m.def("dtoh_memcpy",
          nb::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&dtoh_memcpy<FArrayBox>),
          nb::arg("dest"), nb::arg("src"), nb::arg("scomp"), nb::arg("dcomp"), nb::arg("ncomp"),
          "Copy from a device to host FabArray for a specific (number of) component(s)."
    );

    make_MultiFab(py_MultiFab, "MultiFab");

    m.def("copy_mfab", nb::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Copy), nb::arg("dst"), nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"))
     .def("copy_mfab", nb::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Copy), nb::arg("dst"), nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"));

}
