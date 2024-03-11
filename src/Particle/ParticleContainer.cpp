/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>

#include <cstdint>


namespace
{
    using namespace amrex;

    // Note - this function MUST be consistent with AMReX_Particle.H
    Long unpack_id (uint64_t idcpu) {
        Long r = 0;

        uint64_t sign = idcpu >> 63;  // extract leftmost sign bit
        uint64_t val  = ((idcpu >> 24) & 0x7FFFFFFFFF);  // extract next 39 id bits

        Long lval = static_cast<Long>(val);  // bc we take -
        r = (sign) ? lval : -lval;
        return r;
    }

    // Note - this function MUST be consistent with AMReX_Particle.H
    int unpack_cpu (uint64_t idcpu) {
        return static_cast<int>(idcpu & 0x00FFFFFF);
    }
}

// forward declarations
void init_ParticleContainer_HiPACE(py::module& m);
void init_ParticleContainer_ImpactX(py::module& m);
void init_ParticleContainer_WarpX(py::module& m);

void init_ParticleContainer(py::module& m) {
    using namespace amrex;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects

    // most common case: ND particle + runtime attributes
    //   pure SoA
    make_ParticleContainer_and_Iterators<
        SoAParticle<AMREX_SPACEDIM, 0>,
                    AMREX_SPACEDIM, 0
    >(m);
    //   legacy AoS + SoA
    //make_ParticleContainer_and_Iterators<Particle<0, 0>, 0, 0>(m);

    // used in tests
    make_ParticleContainer_and_Iterators<Particle<2, 1>, 3, 1>(m);

    // application codes
    init_ParticleContainer_HiPACE(m);
    init_ParticleContainer_ImpactX(m);
    init_ParticleContainer_WarpX(m);

    // for particle idcpu arrays
    m.def("unpack_ids", py::vectorize(unpack_id));
    m.def("unpack_cpus", py::vectorize(unpack_cpu));
}
