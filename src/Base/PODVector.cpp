/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_PODVector.H>
#include <AMReX_GpuContainers.H>

#include <algorithm>
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

    std::string
    str_PODVector(std::string typestr, std::string allocstr)
    {
        auto const podv_name = std::string("PODVector_")
            .append(typestr)
            .append("_")
            .append(allocstr);
        return podv_name;
    }

    template <class T, class Allocator>
    bool
    is_host_accessible(PODVector<T, Allocator> const & podvector)
    {
#ifdef AMREX_USE_GPU
        if constexpr (IsArenaAllocator<Allocator>::value) {
            return static_cast<Allocator const &>(podvector)
                .arena()
                ->isHostAccessible();
        } else
#endif
        {
            amrex::ignore_unused(podvector);
            return true;
        }
    }

    template <class PODVector_type>
    std::size_t
    checked_index(PODVector_type const & podvector, int const v)
    {
        auto index = static_cast<std::ptrdiff_t>(v);
        auto const size = static_cast<std::ptrdiff_t>(podvector.size());
        if (index < 0) {
            index += size;
        }
        if (index < 0 || index >= size) {
            throw py::index_error("PODVector index out of range");
        }
        return static_cast<std::size_t>(index);
    }

    template <class T, class Allocator>
    T
    get_item(PODVector<T, Allocator> const & podvector, int const v)
    {
        auto const index = checked_index(podvector, v);
#ifdef AMREX_USE_GPU
        if (!is_host_accessible(podvector)) {
            T value;
            Gpu::copyAsync(
                Gpu::deviceToHost,
                podvector.begin() + index,
                podvector.begin() + index + 1,
                &value
            );
            Gpu::streamSynchronize();
            return value;
        }
#endif
        return podvector[index];
    }

    template <class T, class Allocator>
    void
    set_item(PODVector<T, Allocator> & podvector, int const v, T const value)
    {
        auto const index = checked_index(podvector, v);
#ifdef AMREX_USE_GPU
        if (!is_host_accessible(podvector)) {
            Gpu::copyAsync(
                Gpu::hostToDevice,
                &value,
                &value + 1,
                podvector.begin() + index
            );
            Gpu::streamSynchronize();
            return;
        }
#endif
        podvector[index] = value;
    }

    /** Create a new PODVector with a copy of a host (NumPy) array.
     *
     * Always copies: a ``PODVector`` owns its memory through its allocator,
     * so a zero-copy view is not possible.  The source array is normalized on
     * the host by pybind11 (``c_style | forcecast``): a non-contiguous or
     * differently-typed input is staged into a contiguous, ``T``-typed
     * temporary before the copy.  The destination is filled with a single
     * contiguous transfer: a host copy for host-accessible allocators, or
     * ``Gpu::copyAsync(hostToDevice, ...)`` for device memory.  This keeps
     * the host-to-device path free of any CuPy dependency.
     */
    template <class T, class Allocator>
    PODVector<T, Allocator>
    from_numpy(py::array_t<T, py::array::c_style | py::array::forcecast> const & arr)
    {
        auto const buf = arr.request();
        if (buf.ndim != 1) {
            throw py::value_error("from_numpy: expected a 1-D array");
        }
        auto const n = static_cast<std::size_t>(buf.shape[0]);

        PODVector<T, Allocator> podvector(n);
        if (n == 0) {
            return podvector;
        }

        auto const * src = static_cast<T const *>(buf.ptr);
#ifdef AMREX_USE_GPU
        if (!is_host_accessible(podvector)) {
            Gpu::copyAsync(
                Gpu::hostToDevice,
                src,
                src + n,
                podvector.begin()
            );
            Gpu::streamSynchronize();
            return podvector;
        }
#endif
        std::copy(src, src + n, podvector.begin());
        return podvector;
    }
}

template <class T, class Allocator = std::allocator<T> >
void make_PODVector(py::module &m, std::string typestr, std::string allocstr)
{
    using namespace amrex;

    using PODVector_type = PODVector<T, Allocator>;
    auto const podv_name = str_PODVector(typestr, allocstr);

    auto const podv_doc = std::string(
        "A plain-old-data (POD) vector of '")
        .append(typestr)
        .append("' elements with '")
        .append(allocstr)
        .append("' allocation.");

    py::class_<PODVector_type>(m, podv_name.c_str(), podv_doc.c_str())
        .def("__repr__",
             [typestr](PODVector_type const & pv) {
                 std::stringstream s, rs;
                 s << pv.size();
                 rs << "<amrex.PODVector of type '" + typestr +
                        "' and size '" + s.str() + "'>\n";
                 /* generally not possible, e.g., device arenas:
                 rs << "[ ";
                 for (int ii = 0; ii < int(pv.size()); ii++) {
                     rs << pv[ii] << " ";
                 }
                 rs << "]\n";
                 */
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
        .def("__setitem__", &set_item<T, Allocator>)
        .def("__getitem__", &get_item<T, Allocator>)

        // create a new vector with a copy of a host (NumPy) array
        .def_static("from_numpy", &from_numpy<T, Allocator>,
             py::arg("arr"),
             py::return_value_policy::move,
             R"(Create a new PODVector from a NumPy array (or array-like).

Always copies the data into a newly allocated PODVector. The input is cast to
the vector's element type and made contiguous as needed. The copy into
device-only memory uses an AMReX host-to-device copy and does not require CuPy.

Parameters
----------
arr : array_like
    Input data, convertible to a NumPy array.

Returns
-------
PODVector
    A new PODVector with a copy of the data.
)")
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
    make_PODVector<T, amrex::PolymorphicArenaAllocator<T>> (m, typestr, "polymorphic");

    // Implement AMReX_GpuContainers.H
    // Alias matching Gpu::DeviceVector<T> etc. — resolves per platform:
    //   CPU: PODVector_<type>_std, GPU: PODVector_<type>_arena
    constexpr auto cstr = [](std::string const & a, std::string const & b) { return a + "_" + b; };
#ifdef AMREX_USE_GPU
    m.attr(cstr("DeviceVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "arena").c_str());
    m.attr(cstr("NonManagedDeviceVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "device").c_str());
    m.attr(cstr("ManagedVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "managed").c_str());
    m.attr(cstr("ManagedDeviceVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "managed").c_str());
    m.attr(cstr("PinnedVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "pinned").c_str());
    m.attr(cstr("AsyncVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "async").c_str());
    m.attr(cstr("HostVector", typestr).c_str()) = m.attr(str_PODVector(typestr, "pinned").c_str());
#else
    py::object const std_pod = m.attr(str_PODVector(typestr, "std").c_str());
    m.attr(cstr("DeviceVector", typestr).c_str()) = std_pod;
    m.attr(cstr("NonManagedDeviceVector", typestr).c_str()) = std_pod;
    m.attr(cstr("ManagedVector", typestr).c_str()) = std_pod;
    m.attr(cstr("ManagedDeviceVector", typestr).c_str()) = std_pod;
    m.attr(cstr("PinnedVector", typestr).c_str()) = std_pod;
    m.attr(cstr("AsyncVector", typestr).c_str()) = std_pod;
    m.attr(cstr("HostVector", typestr).c_str()) = std_pod;
#endif

    auto const default_name = str_PODVector(typestr, "default");
    m.attr(default_name.c_str()) =
#ifdef AMREX_USE_GPU
        m.attr(str_PODVector(typestr, "arena").c_str());
#else
        m.attr(str_PODVector(typestr, "std").c_str());
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
