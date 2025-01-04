/* Copyright 2021-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Algorithm.H>

#include <type_traits>


void init_Algorithm (py::module& m)
{
    using namespace amrex;

    m.def(
        "almost_equal",
        &almostEqual<Real>,
        py::arg("x"), py::arg("y"), py::arg("ulp")=2
    );

    if constexpr (!std::is_same_v<Real, ParticleReal>)
    {
        m.def(
            "almost_equal",
            &almostEqual<ParticleReal>,
            py::arg("x"), py::arg("y"), py::arg("ulp")=2
        );
    }
}
