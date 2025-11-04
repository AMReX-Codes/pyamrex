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


void init_MultiFab(py::module &m, py::class_< amrex::MFIter > & py_MFIter)
{
    using namespace amrex;

    py::class_< MultiFab, FabArray<FArrayBox> > py_MultiFab(m, "MultiFab", py::dynamic_attr());

    py_MFIter
        .def("__repr__",
             [](MFIter const & mfi) {
                 std::string r = "<amrex.MFIter (";
                 if( !mfi.isValid() ) { r.append("in"); }
                 r.append("valid)>");
                 return r;
             }
        )
        .def(py::init< FabArrayBase const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the FabArrayBase (argument 2) alive
             py::keep_alive<1, 2>()
        )
        .def(py::init< FabArrayBase const &, MFItInfo const & >())

        .def(py::init< MultiFab const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the MultiFab (argument 2) alive
            py::keep_alive<1, 2>()
        )
        .def(py::init< MultiFab const &, MFItInfo const & >())

        .def(py::init< iMultiFab const & >())
        .def(py::init< iMultiFab const &, MFItInfo const & >())

        // helpers for iteration __next__
        .def("_incr", &MFIter::operator++)
        .def("finalize", &MFIter::Finalize)

        .def("tilebox", py::overload_cast< >(&MFIter::tilebox, py::const_))
        .def("tilebox", py::overload_cast< IntVect const & >(&MFIter::tilebox, py::const_))
        .def("tilebox", py::overload_cast< IntVect const &, IntVect const & >(&MFIter::tilebox, py::const_))

        .def("validbox", &MFIter::validbox)
        .def("fabbox", &MFIter::fabbox)

        .def("nodaltilebox",
            py::overload_cast< int >(&MFIter::nodaltilebox, py::const_),
            py::arg("dir") = -1)

        .def("growntilebox",
            py::overload_cast< const IntVect& >(&MFIter::growntilebox, py::const_),
            py::arg("ng") = -1000000)

        .def("grownnodaltilebox",
            py::overload_cast< int, int >(&MFIter::grownnodaltilebox, py::const_),
            py::arg("int") = -1, py::arg("ng") = -1000000)
        .def("grownnodaltilebox",
            py::overload_cast< int, const IntVect& >(&MFIter::grownnodaltilebox, py::const_),
            py::arg("int"), py::arg("ng"))

        .def_property_readonly("is_valid", &MFIter::isValid)
        .def_property_readonly("index", &MFIter::index)
        .def_property_readonly("length", &MFIter::length)
    ;

    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"),
          "Copy from a host to device FabArray."
    );
    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
          "Copy from a host to device FabArray for a specific (number of) component(s)."
    );

    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"),
          "Copy from a device to host FabArray."
    );
    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
          "Copy from a device to host FabArray for a specific (number of) component(s)."
    );

    make_MultiFab(py_MultiFab, "MultiFab");

    m.def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"))
     .def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"));

}
