/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_ArrayOfStructs.H>
#include <AMReX_GpuAllocators.H>

#include <sstream>


namespace
{
    using namespace amrex;

    // Note - this function MUST be consistent with AMReX_Particle.H
    Long unpack_id (uint64_t cpuid) {
        Long r = 0;

        uint64_t sign = cpuid >> 63;  // extract leftmost sign bit
        uint64_t val  = ((cpuid >> 24) & 0x7FFFFFFFFF);  // extract next 39 id bits

        Long lval = static_cast<Long>(val);  // bc we take -
        r = (sign) ? lval : -lval;
        return r;
    }

    // Note - this function MUST be consistent with AMReX_Particle.H
    int unpack_cpu (uint64_t cpuid) {
        return static_cast<int>(cpuid & 0x00FFFFFF);
    }

    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <typename T_ParticleType,
              template<class> class Allocator=DefaultAllocator>
    py::dict
    array_interface(ArrayOfStructs<T_ParticleType, Allocator> const & aos)
    {
        using ParticleType = T_ParticleType;
        using RealType     = typename ParticleType::RealType;

        auto d = py::dict();
        bool const read_only = false;
        d["data"] = py::make_tuple(std::intptr_t(aos.dataPtr()), read_only);
        d["shape"] = py::make_tuple(aos.size());
        d["strides"] = py::make_tuple(sizeof(ParticleType));
        d["typestr"] = "|V" + std::to_string(sizeof(ParticleType));
        py::list descr;
        descr.append(py::make_tuple("x", py::format_descriptor<RealType>::format()));
#if (AMREX_SPACEDIM >= 2)
        descr.append(py::make_tuple("y", py::format_descriptor<RealType>::format()));
#endif
#if (AMREX_SPACEDIM >= 3)
        descr.append(py::make_tuple("z", py::format_descriptor<RealType>::format()));
#endif
        if constexpr (ParticleType::NReal > 0) {
            for(int ii=0; ii < ParticleType::NReal; ii++) {
                descr.append(py::make_tuple("rdata_"+std::to_string(ii),py::format_descriptor<RealType>::format()));
            }
        }
        descr.append(py::make_tuple("cpuid", py::format_descriptor<uint64_t>::format()) );
        if constexpr (ParticleType::NInt > 0) {
            for(int ii=0; ii < ParticleType::NInt; ++ii) {
                descr.append(py::make_tuple("idata_"+std::to_string(ii),py::format_descriptor<int>::format()));
            }
        }

        d["descr"] = descr;
        d["version"] = 3;
        return d;
    }
}

template <typename T_ParticleType,
          template<class> class Allocator=DefaultAllocator>
void make_ArrayOfStructs(py::module &m, std::string allocstr)
{
    using namespace amrex;

    using AOSType = ArrayOfStructs<T_ParticleType, Allocator>;
    using ParticleType  = T_ParticleType;

    auto const aos_name = std::string("ArrayOfStructs_")
                          .append(std::to_string(ParticleType::NReal)).append("_")
                          .append(std::to_string(ParticleType::NInt)).append("_")
                          .append(allocstr);
    py::class_<AOSType>(m, aos_name.c_str())
        .def(py::init())
        // TODO:
        //operator()
        // .def("__call__", [](AOSType const & pv){ return pv();})
        .def("size", &AOSType::size)
        .def("numParticles", &AOSType::numParticles)
        .def("numRealParticles", &AOSType::numRealParticles)
        .def("numNeighborParticles", &AOSType::numNeighborParticles)
        .def("numTotalParticles", &AOSType::numTotalParticles)
        .def("setNumNeighbors", &AOSType::setNumNeighbors)
        .def("getNumNeighbors", &AOSType::getNumNeighbors)
        .def("empty", py::overload_cast<>(&AOSType::empty))
        .def("empty", py::overload_cast<>(&AOSType::empty, py::const_))
        .def("push_back", &AOSType::push_back)
        .def("pop_back", &AOSType::pop_back)
        .def("back", py::overload_cast<>(&AOSType::back),"get back member.  Problem!!!!! this is perfo")

        // setter & getter
        .def_property_readonly("__array_interface__", [](AOSType const & aos) {
            return array_interface(aos);
        })
        .def_property_readonly("__cuda_array_interface__", [](AOSType const & aos) {
            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            auto d = array_interface(aos);

            // data:
            // Because the user of the interface may or may not be in the same context, the most common case is to use cuPointerGetAttribute with CU_POINTER_ATTRIBUTE_DEVICE_POINTER in the CUDA driver API (or the equivalent CUDA Runtime API) to retrieve a device pointer that is usable in the currently active context.
            // TODO For zero-size arrays, use 0 here.

            // None or integer
            // An optional stream upon which synchronization must take place at the point of consumption, either by synchronizing on the stream or enqueuing operations on the data on the given stream. Integer values in this entry are as follows:
            //   0: This is disallowed as it would be ambiguous between None and the default stream, and also between the legacy and per-thread default streams. Any use case where 0 might be given should either use None, 1, or 2 instead for clarity.
            //   1: The legacy default stream.
            //   2: The per-thread default stream.
            //   Any other integer: a cudaStream_t represented as a Python integer.
            //   When None, no synchronization is required.
            d["stream"] = py::none();

            d["version"] = 3;
            return d;
        })
        .def("test_sizes", [](){ })
        .def("__setitem__", [](AOSType &aos, int const v,  const ParticleType& p){ aos[v] = p; })
        .def("__getitem__", [](AOSType &aos, int const v){ return aos[v]; }, py::return_value_policy::reference)

        .def("to_host", [](AOSType const & aos) {
            ArrayOfStructs<T_ParticleType, std::allocator> h_data;
            h_data.resize(aos.size());
            //py::array_t<T_ParticleType> h_data(aos.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
               aos.begin(), aos.end(),
               h_data.begin()
               //h_data.ptr()
            );
            return h_data;
        })
    ;
}

template <int NReal, int NInt>
void make_ArrayOfStructs(py::module &m)
{
    using namespace amrex;

    // AMReX legacy AoS position + id/cpu particle ype
    using ParticleType = Particle<NReal, NInt>;

    // see Src/Base/AMReX_GpuContainers.H
    //   !AMREX_USE_GPU: DefaultAllocator = std::allocator
    //    AMREX_USE_GPU: DefaultAllocator = amrex::ArenaAllocator

    //   work-around for https://github.com/pybind/pybind11/pull/4581
    //make_ArrayOfStructs<ParticleType, std::allocator> (m, "std");
    //make_ArrayOfStructs<ParticleType, amrex::ArenaAllocator> (m, "arena");
#ifdef AMREX_USE_GPU
    make_ArrayOfStructs<ParticleType, std::allocator> (m, "std");
    make_ArrayOfStructs<ParticleType, amrex::DefaultAllocator> (m, "default");  // amrex::ArenaAllocator
#else
    make_ArrayOfStructs<ParticleType, amrex::DefaultAllocator> (m, "default");  // std::allocator
    make_ArrayOfStructs<ParticleType, amrex::ArenaAllocator> (m, "arena");
#endif
    //   end work-around
    make_ArrayOfStructs<ParticleType, amrex::PinnedArenaAllocator> (m, "pinned");
#ifdef AMREX_USE_GPU
    make_ArrayOfStructs<ParticleType, amrex::DeviceArenaAllocator> (m, "device");
    make_ArrayOfStructs<ParticleType, amrex::ManagedArenaAllocator> (m, "managed");
    make_ArrayOfStructs<ParticleType, amrex::AsyncArenaAllocator> (m, "async");
#endif
}

void init_ArrayOfStructs(py::module& m) {
    make_ArrayOfStructs<0, 0> (m);  // WarpX 22.07, ImpactX 22.07, HiPACE++ 22.07
    make_ArrayOfStructs<1, 1> (m);  // test in ParticleContainer
    make_ArrayOfStructs<2, 1> (m);  // test

    m.def("unpack_ids", py::vectorize(unpack_id));
    m.def("unpack_cpus", py::vectorize(unpack_cpu));
}
