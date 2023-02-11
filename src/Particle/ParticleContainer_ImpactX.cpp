/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"


void init_ParticleContainer_ImpactX(py::module& m) {
    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects
    make_ParticleContainer_and_Iterators< 0, 0, 5, 0> (m);   // ImpactX 22.07
}
