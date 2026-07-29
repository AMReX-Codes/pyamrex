/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"
#include "MultiFab.H"

#include <AMReX_FabArray.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>


void init_iMultiFab(nb::module_ &m)
{
    using namespace amrex;

    nb::class_< iMultiFab, FabArray<IArrayBox> > py_iMultiFab(m, "iMultiFab");
    make_MultiFab(py_iMultiFab, "iMultiFab");

    m.def("copy_mfab", nb::overload_cast< iMultiFab &, iMultiFab const &, int, int, int, int >(&iMultiFab::Copy), nb::arg("dst"), nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"))
     .def("copy_mfab", nb::overload_cast< iMultiFab &, iMultiFab const &, int, int, int, IntVect const & >(&iMultiFab::Copy), nb::arg("dst"), nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"));

    // host-device copies for integer FabArrays (iMultiFab)
    m.def("htod_memcpy",
          nb::overload_cast< FabArray<IArrayBox> &, FabArray<IArrayBox> const & >(&htod_memcpy<IArrayBox>),
          nb::arg("dest"), nb::arg("src"),
          "Copy from a host to device FabArray."
    );
    m.def("htod_memcpy",
          nb::overload_cast< FabArray<IArrayBox> &, FabArray<IArrayBox> const &, int, int, int >(&htod_memcpy<IArrayBox>),
          nb::arg("dest"), nb::arg("src"), nb::arg("scomp"), nb::arg("dcomp"), nb::arg("ncomp"),
          "Copy from a host to device FabArray for a specific (number of) component(s)."
    );

    m.def("dtoh_memcpy",
          nb::overload_cast< FabArray<IArrayBox> &, FabArray<IArrayBox> const & >(&dtoh_memcpy<IArrayBox>),
          nb::arg("dest"), nb::arg("src"),
          "Copy from a device to host FabArray."
    );
    m.def("dtoh_memcpy",
          nb::overload_cast< FabArray<IArrayBox> &, FabArray<IArrayBox> const &, int, int, int >(&dtoh_memcpy<IArrayBox>),
          nb::arg("dest"), nb::arg("src"), nb::arg("scomp"), nb::arg("dcomp"), nb::arg("ncomp"),
          "Copy from a device to host FabArray for a specific (number of) component(s)."
    );
}
