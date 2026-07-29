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

    nb::object pack_ids (nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>> idcpus,
                         nb::ndarray<nb::numpy, amrex::Long, nb::ndim<1>> ids)
    {
        if (idcpus.ndim() != 1) {
            throw std::runtime_error("Input should be 1-D NumPy array");
        }

        if (idcpus.size() != ids.size()) {
            throw std::runtime_error("sizes do not match!");
        }

        int N = static_cast<int>(idcpus.shape(0));
        auto * idcpus_ptr = idcpus.data();
        auto * ids_ptr = ids.data();
        for (int i = 0; i < N; i++) {
            particle_impl::pack_id(idcpus_ptr[i], ids_ptr[i]);
        }
        return nb::none();
    }

    nb::object pack_cpus (nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>> idcpus,
                          nb::ndarray<nb::numpy, int, nb::ndim<1>> cpus)
    {
        if (idcpus.ndim() != 1) {
            throw std::runtime_error("Input should be 1-D NumPy array");
        }

        if (idcpus.size() != cpus.size()) {
            throw std::runtime_error("sizes do not match!");
        }

        int N = static_cast<int>(idcpus.shape(0));
        auto * idcpus_ptr = idcpus.data();
        auto * cpus_ptr = cpus.data();
        for (int i = 0; i < N; i++) {
            particle_impl::pack_cpu(idcpus_ptr[i], cpus_ptr[i]);
        }
        return nb::none();
    }

    Long unpack_id (uint64_t idcpu) {
        return particle_impl::unpack_id(idcpu);
    }

    int unpack_cpu (uint64_t idcpu) {
        return particle_impl::unpack_cpu(idcpu);
    }

    template<typename T, T (*Unpack)(uint64_t)>
    nb::object unpack_array (
        nb::ndarray<nb::numpy, uint64_t, nb::c_contig> idcpus)
    {
        nb::list shape;
        for (std::size_t i = 0; i < idcpus.ndim(); ++i)
            shape.append(idcpus.shape(i));

        auto numpy = nb::module_::import_("numpy");
        auto result = numpy.attr("empty")(
            shape, numpy.attr("dtype")(pyAMReX::buffer_format<T>())
        );
        auto output = nb::cast<nb::ndarray<nb::numpy, T, nb::c_contig>>(result);
        auto const * input_ptr = idcpus.data();
        auto * output_ptr = output.data();
        for (std::size_t i = 0; i < idcpus.size(); ++i)
            output_ptr[i] = Unpack(input_ptr[i]);
        return result;
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
#ifdef PYAMREX_CODES_SoA
void init_ParticleContainer_SoA(nb::module_& m);
#endif
#ifdef PYAMREX_CODES_tests
void init_ParticleContainer_tests(nb::module_& m);
#endif
#ifdef PYAMREX_CODES_FHDeX
void init_ParticleContainer_FHDeX(nb::module_& m);
#endif
#ifdef PYAMREX_CODES_ImpactX
void init_ParticleContainer_ImpactX(nb::module_& m);
#endif
#ifdef PYAMREX_CODES_WarpX
void init_ParticleContainer_WarpX(nb::module_& m);
#endif

void init_ParticleContainer(nb::module_& m) {
    using namespace amrex;

    // TODO: we might need to move all or most of the defines in here into a
    //       test/example submodule, so they do not collide with downstream projects

    // most common case: ND particle + runtime attributes
    //   pure SoA
#ifdef PYAMREX_CODES_SoA
    init_ParticleContainer_SoA(m);
#endif
    //   legacy AoS + SoA
    //make_ParticleContainer_and_Iterators<Particle<0, 0>, 0, 0>(m);

    // used in tests
#ifdef PYAMREX_CODES_tests
    init_ParticleContainer_tests(m);
#endif

    // application codes
#ifdef PYAMREX_CODES_FHDeX
    init_ParticleContainer_FHDeX(m);
#endif
#ifdef PYAMREX_CODES_ImpactX
    init_ParticleContainer_ImpactX(m);
#endif
#ifdef PYAMREX_CODES_WarpX
    init_ParticleContainer_WarpX(m);
#endif

    // for particle idcpu arrays
    m.def("pack_ids", &pack_ids);
    m.def("pack_cpus", &pack_cpus);
    m.def("unpack_ids", &unpack_id);
    m.def("unpack_ids", &unpack_array<Long, unpack_id>);
    m.def("unpack_cpus", &unpack_cpu);
    m.def("unpack_cpus", &unpack_array<int, unpack_cpu>);
    m.def("make_invalid", make_invalid);
    m.def("make_valid", make_valid);
    m.def("is_valid", is_valid);
}
