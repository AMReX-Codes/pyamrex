/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"


void init_Array4_float(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< float >(m, "float");
    make_Array4< double >(m, "double");
    make_Array4< long double >(m, "longdouble");
}
