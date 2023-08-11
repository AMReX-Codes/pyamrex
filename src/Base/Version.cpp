/* Copyright 2021-2023 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Axel Huebl
 */
#include "pyAMReX.H"

#include <AMReX.H>
#include <AMReX_Version.H>


void init_Version (py::module& m)
{
    // API runtime version
    //   note PEP-440 syntax: x.y.zaN but x.y.z.devN
    m.attr("__version__") = amrex::Version();
}
