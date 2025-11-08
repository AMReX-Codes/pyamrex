/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_PODVector.H>
#include <AMReX_GpuContainers.H>

#include <sstream>


namespace
{
    using namespace amrex;

    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <class T, class Allocator = std::allocator<T> >
    py::dict
    array_interface(PODVector<T, Allocator> const & podvector)
    {
        auto d = py::dict();
        bool const read_only = false;
        d["data"] = py::make_tuple(std::intptr_t(podvector.dataPtr()), read_only);
        d["shape"] = py::make_tuple(podvector.size());
        d["strides"] = py::none();
        d["typestr"] = py::format_descriptor<T>::format();
        d["version"] = 3;
        return d;
    }
}

template <class T, class Allocator = std::allocator<T> >
void make_PODVector(py::module &m, std::string typestr, std::string allocstr)
{
    using namespace amrex;

    using PODVector_type = PODVector<T, Allocator>;
    auto const podv_name = std::string("PODVector_").append(typestr)
                           .append("_").append(allocstr);

    py::class_<PODVector_type>(m, podv_name.c_str())
        .def("__repr__",
             [typestr](PODVector_type const & pv) {
                 std::stringstream s, rs;
                 s << pv.size();
                 rs << "<amrex.PODVector of type '" + typestr +
                        "' and size '" + s.str() + "'>\n";
                 rs << "[ ";
                 for (int ii = 0; ii < int(pv.size()); ii++) {
                     rs << pv[ii] << " ";
                 }
                 rs << "]\n";
                 return rs.str();
             }
        )
        .def(py::init<>())
        .def(py::init<std::size_t>(), py::arg("size"))
        .def(py::init<PODVector_type&>(), py::arg("other"))
        .def("assign", [](PODVector_type & pv, T const & value){
            pv.assign(pv.size(), value);
        }, py::arg("value"), "assign the same value to every element")
        .def("push_back", py::overload_cast<const T&>(&PODVector_type::push_back))
        .def("pop_back", &PODVector_type::pop_back)
        .def("clear", &PODVector_type::clear)
        .def("size", &PODVector_type::size)
        .def("__len__", &PODVector_type::size)
        // .def("max_size", &PODVector_type::max_size)
        .def("capacity", &PODVector_type::capacity)
        .def("empty", &PODVector_type::empty)
        .def("resize",
            py::overload_cast<std::size_t, GrowthStrategy>(&PODVector_type::resize),
            py::arg("new_size"),
            py::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("resize",
            py::overload_cast<std::size_t, const T&, GrowthStrategy>(&PODVector_type::resize),
            py::arg("new_size"),
            py::arg("value"),
            py::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("reserve",
            &PODVector_type::reserve,
            py::arg("capacity"),
            py::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("shrink_to_fit", &PODVector_type::shrink_to_fit)
        .def("to_host", [](PODVector_type const & pv) {
            PODVector<T, amrex::PinnedArenaAllocator<T>> h_data(pv.size());
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
               pv.begin(), pv.end(),
               h_data.begin()
            );
            Gpu::streamSynchronize();
            return h_data;
        }, py::return_value_policy::move)

        // front
        // back
        // data
        // begin
        //
        // swap

        .def_property_readonly("__array_interface__", [](PODVector_type const & podvector) {
            return array_interface(podvector);
        })
        .def_property_readonly("__cuda_array_interface__", [](PODVector_type const & podvector) {
            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            auto d = array_interface(podvector);

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
        // setter & getter
        .def("__setitem__", [](PODVector_type & podvector, int const v, T const value){ podvector[v] = value; })
        .def("__getitem__", [](PODVector_type & pv, int const v){ return pv[v]; })
    ;
}

template <class T>
void make_PODVector(py::module &m, std::string typestr)
{
    // see Src/Base/AMReX_GpuContainers.H
    make_PODVector<T, amrex::PinnedArenaAllocator<T>> (m, typestr, "pinned");
    make_PODVector<T, amrex::ArenaAllocator<T>> (m, typestr, "arena");
    make_PODVector<T, std::allocator<T>> (m, typestr, "std");
#ifdef AMREX_USE_GPU
    make_PODVector<T, amrex::DeviceArenaAllocator<T>> (m, typestr, "device");
    make_PODVector<T, amrex::ManagedArenaAllocator<T>> (m, typestr, "managed");
    make_PODVector<T, amrex::AsyncArenaAllocator<T>> (m, typestr, "async");
#endif
}

void init_PODVector(py::module& m)
{
    py::native_enum<GrowthStrategy>(m, "GrowthStrategy", "enum.Enum")
        .value("Poisson", GrowthStrategy::Poisson)
        .value("Exact", GrowthStrategy::Exact)
        .value("Geometric", GrowthStrategy::Geometric)
        .export_values()
        .finalize()
    ;

    make_PODVector<ParticleReal> (m, "real");
    make_PODVector<int> (m, "int");
    make_PODVector<uint64_t> (m, "uint64");
}
