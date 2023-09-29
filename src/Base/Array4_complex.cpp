/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

#include <complex>


void init_Array4_complex(py::module &m)
{
    using namespace pyAMReX;

    make_Array4< std::complex<float> >(m, "cfloat");
    make_Array4< std::complex<double> >(m, "cdouble");

    // not great on device
    //   NVCC Warning #20208-D: 'long double' is treated as 'double' in device code
    //make_Array4< std::complex<long double> >(m, "clongdouble");
}
