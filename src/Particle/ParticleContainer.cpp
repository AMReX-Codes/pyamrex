/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl, Andrew Myers
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleContainer.H"

#include <AMReX_Particle.H>

#include <cstdint>


namespace
{
    using namespace amrex;

    py::object pack_ids (py::array_t<uint64_t> idcpus,
                         py::array_t<amrex::Long> ids)
    {
        if (idcpus.ndim() != 1) {
            throw std::runtime_error("Input should be 1-D NumPy array");
        }

        auto buf = idcpus.request();
        auto buf2 = ids.request();
        if (buf.size != buf2.size) {
            throw std::runtime_error("sizes do not match!");
        }

        int N = idcpus.shape()[0];
        for (int i = 0; i < N; i++) {
            uint64_t* idcpus_ptr = (uint64_t*) buf.ptr;
            amrex::Long* ids_ptr = (amrex::Long*) buf2.ptr;
            particle_impl::pack_id(idcpus_ptr[i], ids_ptr[i]);
        }
        return py::cast<py::none>(Py_None);
    }

    py::object pack_cpus (py::array_t<uint64_t> idcpus,
                          py::array_t<int> cpus)
    {
        if (idcpus.ndim() != 1) {
            throw std::runtime_error("Input should be 1-D NumPy array");
        }

        auto buf = idcpus.request();
        auto buf2 = cpus.request();
        if (buf.size != buf2.size) {
            throw std::runtime_error("sizes do not match!");
        }

        int N = idcpus.shape()[0];
        for (int i = 0; i < N; i++) {
            uint64_t* idcpus_ptr = (uint64_t*) buf.ptr;
            int* cpus_ptr = (int*) buf2.ptr;
            particle_impl::pack_cpu(idcpus_ptr[i], cpus_ptr[i]);
        }
        return py::cast<py::none>(Py_None);
    }

    Long unpack_id (uint64_t idcpu) {
        return particle_impl::unpack_id(idcpu);
    }

    int unpack_cpu (uint64_t idcpu) {
        return particle_impl::unpack_cpu(idcpu);
    }

    uint64_t make_invalid (uint64_t idcpu) {
        particle_impl::make_invalid(idcpu);
        return idcpu;
    }

    uint64_t make_valid (uint64_t idcpu) {
        particle_impl::make_valid(idcpu);
        return idcpu;
    }

    bool is_valid (const uint64_t idcpu) {
        return particle_impl::is_valid(idcpu);
    }
}

// forward declarations
void init_ParticleContainer_FHDeX(py::module& m);
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
    init_ParticleContainer_FHDeX(m);
    init_ParticleContainer_ImpactX(m);
    init_ParticleContainer_WarpX(m);

    // for particle idcpu arrays
    m.def("pack_ids", &pack_ids);
    m.def("pack_cpus", &pack_cpus);
    m.def("unpack_ids", py::vectorize(unpack_id));
    m.def("unpack_cpus", py::vectorize(unpack_cpu));
    m.def("make_invalid", make_invalid);
    m.def("make_valid", make_valid);
    m.def("is_valid", is_valid);
}
