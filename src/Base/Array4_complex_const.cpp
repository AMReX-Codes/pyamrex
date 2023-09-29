/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <complex>


void init_Array4_complex_const(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< std::complex<float> const >(m, "cfloat_const");
    make_Array4< std::complex<double> const >(m, "cdouble_const");

    // not great on device:
    //   NVCC Warning #20208-D: 'long double' is treated as 'double' in device code
    //make_Array4< std::complex<long double const> >(m, "clongdouble_const");
}
