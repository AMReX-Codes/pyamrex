/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>


void init_ParticleContainer_soa(py::module& m) {
    using namespace amrex;

    // most common case: ND particle + runtime attributes
    //   pure SoA
    make_ParticleContainer_and_Iterators<
        SoAParticle<AMREX_SPACEDIM, 0>,
                    AMREX_SPACEDIM, 0
    >(m);
}
