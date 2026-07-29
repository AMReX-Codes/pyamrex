/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>


void init_ParticleContainer_tests(nb::module_& m) {
    using namespace amrex;

    make_ParticleContainer_and_Iterators<Particle<2, 1>, 3, 1>(m);
}
