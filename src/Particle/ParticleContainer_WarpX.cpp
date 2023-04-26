/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>


void init_ParticleContainer_WarpX(py::module& m) {
    using namespace amrex;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleContainer_and_Iterators<Particle<0, 0>, 4, 0>(m);   // WarpX 22.07 - 23.04 1D-3D
    //make_ParticleContainer_and_Iterators<Particle<0, 0>, 5, 0> (m);   // WarpX 22.07 - 23.04 RZ
}
