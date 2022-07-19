/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_ArrayOfStructs.H>

#include <sstream>

namespace py = pybind11;
using namespace amrex;


template <int NReal, int NInt,
          template<class> class Allocator=DefaultAllocator>
void make_ArrayOfStructs(py::module &m)
{
    using AOSType = ArrayOfStructs<NReal, NInt>;
    using ParticleType  = Particle<NReal, NInt>;
    using RealType      = typename ParticleType::RealType;

    auto const aos_name = std::string("ArrayOfStructs_").append(std::to_string(NReal) + "_" + std::to_string(NInt));
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
        })
        .def("test_sizes", [](){ })
        .def("__setitem__", [](AOSType &aos, int const v,  const ParticleType& p){ aos[v] = p; })
        .def("__getitem__", [](AOSType &aos, int const v){ return aos[v]; }, py::return_value_policy::reference)
    ;
}

void init_ArrayOfStructs(py::module& m) {
    make_ArrayOfStructs<7,0> (m);
    make_ArrayOfStructs< 1, 1> (m);
    make_ArrayOfStructs< 2, 1> (m);
}
