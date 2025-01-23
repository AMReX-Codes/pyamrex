/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_Box.H>
#include <AMReX_Vector.H>

#include <string>
#include <type_traits>


void init_Vector(py::module& m)
{
    using namespace amrex;

    make_Vector<Real> (m, "Real");
    if constexpr(!std::is_same_v<Real, ParticleReal>)
        make_Vector<ParticleReal> (m, "ParticleReal");

    make_Vector<int> (m, "int");
    if constexpr(!std::is_same_v<int, Long>)
        make_Vector<Long> (m, "Long");

    make_Vector<Box> (m, "Box");
    make_Vector<std::string> (m, "string");
}
