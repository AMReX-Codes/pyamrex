/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <cstdint>


void init_Array4_int_const(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< short const >(m, "short_const");
    make_Array4< int const >(m, "int_const");
    make_Array4< long const >(m, "long_const");
    make_Array4< long long const >(m, "longlong_const");
}
