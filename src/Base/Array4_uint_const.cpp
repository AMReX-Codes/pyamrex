/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <cstdint>


void init_Array4_uint_const(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< unsigned short const >(m, "ushort_const");
    make_Array4< unsigned int const >(m, "uint_const");
    make_Array4< unsigned long const >(m, "ulong_const");
    make_Array4< unsigned long long const >(m, "ulonglong_const");
}
