/* Copyright 2021-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "SmallMatrix.H"


void init_SmallMatrix (py::module &m)
{
    using namespace pyAMReX;

    // 6x6 Matrix as commonly used in accelerator physics
    {
        constexpr int NRows = 6;
        constexpr int NCols = 6;
        constexpr amrex::Order ORDER = amrex::Order::F;
        constexpr int StartIndex = 1;

        make_SmallMatrix< float, NRows, NCols, ORDER, StartIndex >(m, "float");
        make_SmallMatrix< double, NRows, NCols, ORDER, StartIndex >(m, "double");
        make_SmallMatrix< long double, NRows, NCols, ORDER, StartIndex >(m, "longdouble");
        /*
        make_SmallMatrix< float const, NRows, NCols, ORDER, StartIndex >(m, "float_const");
        make_SmallMatrix< double const, NRows, NCols, ORDER, StartIndex >(m, "double_const");
        make_SmallMatrix< long double const, NRows, NCols, ORDER, StartIndex >(m, "longdouble_const");
        */
    }
}
