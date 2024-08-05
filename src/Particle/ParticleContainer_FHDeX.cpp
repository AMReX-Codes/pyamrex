/* Copyright 2024 The AMReX Community
 *
 * Authors: Johannes Blaschke
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>


void init_ParticleContainer_FHDeX(py::module& m) {
    using namespace amrex;

    make_ParticleContainer_and_Iterators<Particle<16, 4>, 0, 0>(m);
}
