/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <cstdint>


void init_Array4_uint(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< unsigned short >(m, "ushort");
    make_Array4< unsigned int >(m, "uint");
    make_Array4< unsigned long >(m, "ulong");
    make_Array4< unsigned long long >(m, "ulonglong");
}
