/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"


void init_Array4_float_const(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< float const >(m, "float_const");
    make_Array4< double const >(m, "double_const");
    make_Array4< long double const >(m, "longdouble_const");
}
