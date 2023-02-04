/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_ArrayOfStructs.H>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;
using namespace amrex;

namespace
{
    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <int NReal, int NInt,
              template<class> class Allocator=DefaultAllocator>
    py::dict
    array_interface(ArrayOfStructs<NReal, NInt, Allocator> const & aos)
    {
        using ParticleType = Particle<NReal, NInt>;
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
        if (NReal > 0) {
            for(int ii=0; ii < NReal; ii++) {
                descr.append(py::make_tuple("rdata_"+std::to_string(ii),py::format_descriptor<RealType>::format()));
            }
        }
        descr.append(py::make_tuple("cpuid", py::format_descriptor<uint64_t>::format()) );
        if (NInt > 0) {
            for(int ii=0; ii < NInt; ++ii) {
                descr.append(py::make_tuple("idata_"+std::to_string(ii),py::format_descriptor<int>::format()));
            }
        }

        d["descr"] = descr;
        d["version"] = 3;
        return d;
    }
}

template <int NReal, int NInt,
          template<class> class Allocator=DefaultAllocator>
void make_ArrayOfStructs(py::module &m, std::string allocstr)
{
    using AOSType = ArrayOfStructs<NReal, NInt, Allocator>;
    using ParticleType  = Particle<NReal, NInt>;

    auto const aos_name = std::string("ArrayOfStructs_")
                          .append(std::to_string(NReal)).append("_")
                          .append(std::to_string(NInt)).append("_")
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
    ;
}

template <int NReal, int NInt>
void make_ArrayOfStructs(py::module &m)
{
    // see Src/Base/AMReX_GpuContainers.H
    make_ArrayOfStructs<NReal, NInt, std::allocator> (m, "std");
    make_ArrayOfStructs<NReal, NInt, amrex::ArenaAllocator> (m, "arena");
    make_ArrayOfStructs<NReal, NInt, amrex::DeviceArenaAllocator> (m, "device");
    make_ArrayOfStructs<NReal, NInt, amrex::ManagedArenaAllocator> (m, "managed");
    make_ArrayOfStructs<NReal, NInt, amrex::PinnedArenaAllocator> (m, "pinned");
    make_ArrayOfStructs<NReal, NInt, amrex::AsyncArenaAllocator> (m, "async");
}

void init_ArrayOfStructs(py::module& m) {
    make_ArrayOfStructs<0, 0> (m);  // WarpX 22.07, ImpactX 22.07, HiPACE++ 22.07
    make_ArrayOfStructs<1, 1> (m);  // test in ParticleContainer
    make_ArrayOfStructs<2, 1> (m);  // test
}
