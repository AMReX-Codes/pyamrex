/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <cstdint>


void init_Array4_int(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< short >(m, "short");
    make_Array4< int >(m, "int");
    make_Array4< long >(m, "long");
    make_Array4< long long >(m, "longlong");
}
