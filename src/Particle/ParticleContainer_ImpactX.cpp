/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>


void init_ParticleContainer_ImpactX(py::module& m) {
    using namespace amrex;

    constexpr bool only_polymorphic = true;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleContainer_and_Iterators<SoAParticle<11, 0, ParticleReal>, 11, 0, only_polymorphic>(m);  // ImpactX 26.01+ (native ParticleReal precision)
#ifndef AMREX_SINGLE_PRECISION_PARTICLES
    // additionally provide a single-precision storage container for runtime
    // precision selection (named with a "_sp" suffix); when particles are
    // already single precision the native registration above already covers it
    make_ParticleContainer_and_Iterators<SoAParticle<11, 0, float>, 11, 0, only_polymorphic>(m);  // ImpactX single precision
#endif
}
