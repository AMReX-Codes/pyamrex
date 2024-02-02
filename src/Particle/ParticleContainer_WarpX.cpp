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
#if AMREX_SPACEDIM == 1
    make_ParticleContainer_and_Iterators<SoAParticle<5, 0>, 5, 0>(m);  // WarpX 24.03+ 1D
#elif AMREX_SPACEDIM == 2
    make_ParticleContainer_and_Iterators<SoAParticle<6, 0>, 6, 0>(m);  // WarpX 24.03+ 2D
    make_ParticleContainer_and_Iterators<SoAParticle<7, 0>, 7, 0>(m);  // WarpX 24.03+ RZ
#elif AMREX_SPACEDIM == 3
    make_ParticleContainer_and_Iterators<SoAParticle<7, 0>, 7, 0>(m);  // WarpX 24.03+ 3D
#endif
}
