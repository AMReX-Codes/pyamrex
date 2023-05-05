/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>


void init_ParticleContainer_HiPACE(py::module_& m);
void init_ParticleContainer_ImpactX(py::module_& m);
void init_ParticleContainer_WarpX(py::module_& m);

void init_ParticleContainer(py::module_& m) {
    using namespace amrex;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleContainer_and_Iterators<Particle<1, 1>, 2, 1>(m);  // tests

    init_ParticleContainer_HiPACE(m);
    init_ParticleContainer_ImpactX(m);
    init_ParticleContainer_WarpX(m);
}
