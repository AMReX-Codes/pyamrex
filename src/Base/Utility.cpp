/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Revathi Jambunathan, Axel Huebl
 */
#include "pyAMReX.H"

#include <AMReX_Utility.H>

#include <string>


void init_Utility(nb::module_& m)
{
    m.def("concatenate",
          &amrex::Concatenate,
          "Builds plotfile name",
          nb::arg("root"), nb::arg("num"), nb::arg("mindigits")=5
    );
}
