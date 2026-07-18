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
    nb::dict
    array_interface(PODVector<T, Allocator> const & podvector)
    {
        auto d = nb::dict();
        bool const read_only = false;
        d["data"] = nb::make_tuple(std::intptr_t(podvector.dataPtr()), read_only);
        d["shape"] = nb::make_tuple(podvector.size());
        d["strides"] = nb::none();
        d["typestr"] = pyAMReX::buffer_format<T>();
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
            throw nb::index_error("PODVector index out of range");
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
     * the host by nanobind (``c_contig``): a non-contiguous or
     * differently-typed input is staged into a contiguous, ``T``-typed
     * temporary before the copy.  The destination is filled with a single
     * contiguous transfer: a host copy for host-accessible allocators, or
     * ``Gpu::copyAsync(hostToDevice, ...)`` for device memory.  This keeps
     * the host-to-device path free of any CuPy dependency.
     */
    template <class T, class Allocator>
    PODVector<T, Allocator>
    from_numpy(nb::object const& input)
    {
        auto np = nb::module_::import_("numpy");
        auto const dtype = pyAMReX::buffer_format<T>();
        auto normalized = np.attr("asarray")(
            input,
            nb::arg("dtype") = nb::str(dtype.c_str()),
            nb::arg("order") = "C");
        auto arr = nb::cast<
            nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig>
        >(normalized);
        auto const n = static_cast<std::size_t>(arr.shape(0));

        PODVector<T, Allocator> podvector(n);
        if (n == 0) {
            return podvector;
        }

        auto const * src = arr.data();
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
nb::class_<PODVector<T, Allocator> >
make_PODVector(nb::module_ &m, std::string typestr, std::string allocstr) {
    using namespace amrex;

    using PODVector_type = PODVector<T, Allocator>;
    auto const podv_name = str_PODVector(typestr, allocstr);

    auto const podv_doc = std::string(
        "A plain-old-data (POD) vector of '")
        .append(typestr)
        .append("' elements with '")
        .append(allocstr)
        .append("' allocation.");

    auto cl = nb::class_<PODVector_type>(m, podv_name.c_str(), podv_doc.c_str())
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
        .def(nb::init<>())
        .def("__init__", [](PODVector_type *self, std::size_t size) {
            new (self) PODVector_type(size);
        }, nb::arg("size"))
        .def(nb::init<PODVector_type&>(), nb::arg("other"))
        .def("assign", [](PODVector_type & pv, T const & value){
            pv.assign(pv.size(), value);
        }, nb::arg("value"), "assign the same value to every element")
        .def("push_back", nb::overload_cast<const T&>(&PODVector_type::push_back))
        .def("pop_back", &PODVector_type::pop_back)
        .def("clear", &PODVector_type::clear)
        .def("size", &PODVector_type::size)
        .def("__len__", &PODVector_type::size)
        // .def("max_size", &PODVector_type::max_size)
        .def("capacity", &PODVector_type::capacity)
        .def("empty", &PODVector_type::empty)
        .def("resize",
            nb::overload_cast<std::size_t, GrowthStrategy>(&PODVector_type::resize),
            nb::arg("new_size"),
            nb::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("resize",
            nb::overload_cast<std::size_t, const T&, GrowthStrategy>(&PODVector_type::resize),
            nb::arg("new_size"),
            nb::arg("value"),
            nb::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("reserve",
            &PODVector_type::reserve,
            nb::arg("capacity"),
            nb::arg("strategy") = GrowthStrategy::Poisson
        )
        .def("shrink_to_fit", &PODVector_type::shrink_to_fit)

        // front
        // back
        // data
        // begin
        //
        // swap

        .def_prop_ro("__array_interface__", [](PODVector_type const & podvector) {
            return array_interface(podvector);
        })
        .def_prop_ro("__cuda_array_interface__", [](PODVector_type const & podvector) {
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
            d["stream"] = nb::none();

            d["version"] = 3;
            return d;
        })
        // setter & getter
        .def("__setitem__", &set_item<T, Allocator>)
        .def("__getitem__", &get_item<T, Allocator>)

        // create a new vector with a copy of a host (NumPy) array
        .def_static("from_numpy", &from_numpy<T, Allocator>,
             nb::arg("arr"),
             nb::rv_policy::move,
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

    return cl;
}

/** Bind the host/device copy helpers ``to_host`` and ``to_device`` on a
 * single PODVector class.
 *
 * Both return a different PODVector allocator type than ``cl`` (``to_host`` a
 * pinned vector, ``to_device`` a ``Gpu::DeviceVector`` = ``arena`` on GPU,
 * ``std`` on CPU). Binding them only after every allocator class exists lets
 * nanobind's stub generator resolves those return types to registered Python types.
 */
template <class T, class Allocator>
void bind_host_device(nb::class_<PODVector<T, Allocator> > & cl)
{
    using namespace amrex;

    cl.def("to_host",
        [](PODVector<T, Allocator> const & src) {
            PODVector<T, amrex::PinnedArenaAllocator<T>> h_data(src.size());
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
               src.begin(), src.end(),
               h_data.begin()
            );
            Gpu::streamSynchronize();
            return h_data;
        },
        nb::rv_policy::move,
        "Copy this vector into a new pinned (host) PODVector. Mirrors to_device().")

      .def("to_device",
        [](PODVector<T, Allocator> const & src) -> amrex::Gpu::DeviceVector<T> {
            amrex::Gpu::DeviceVector<T> dst(src.size());
            if (src.empty()) {
                return dst;
            }
#ifdef AMREX_USE_GPU
            bool const src_host = is_host_accessible(src);
            bool const dst_host = is_host_accessible(dst);
            if (src_host && !dst_host) {
                Gpu::copyAsync(Gpu::hostToDevice, src.begin(), src.end(), dst.begin());
                Gpu::streamSynchronize();
                return dst;
            } else if (!src_host && dst_host) {
                Gpu::copyAsync(Gpu::deviceToHost, src.begin(), src.end(), dst.begin());
                Gpu::streamSynchronize();
                return dst;
            } else if (!src_host && !dst_host) {
                Gpu::copyAsync(Gpu::deviceToDevice, src.begin(), src.end(), dst.begin());
                Gpu::streamSynchronize();
                return dst;
            }
#endif
            std::copy(src.begin(), src.end(), dst.begin());
            return dst;
        },
        nb::rv_policy::move,
        "Copy this vector into a new amrex Gpu::DeviceVector (the arena "
        "allocator on GPU, std on CPU), transferring across memory spaces "
        "as needed. Mirrors to_host().");
}

/** Bind ``to_host``/``to_device`` on each of the given PODVector classes. */
template <class... PODVectorClass>
void add_host_device(PODVectorClass &... cls)
{
    (bind_host_device(cls), ...);
}

template <class T>
void make_PODVector(nb::module_ &m, std::string typestr)
{
    // see Src/Base/AMReX_GpuContainers.H
    auto pv_pinned = make_PODVector<T, amrex::PinnedArenaAllocator<T>> (m, typestr, "pinned");
    auto pv_arena = make_PODVector<T, amrex::ArenaAllocator<T>> (m, typestr, "arena");
    auto pv_std = make_PODVector<T, std::allocator<T>> (m, typestr, "std");
#ifdef AMREX_USE_GPU
    auto pv_device = make_PODVector<T, amrex::DeviceArenaAllocator<T>> (m, typestr, "device");
    auto pv_managed = make_PODVector<T, amrex::ManagedArenaAllocator<T>> (m, typestr, "managed");
    auto pv_async = make_PODVector<T, amrex::AsyncArenaAllocator<T>> (m, typestr, "async");
#endif
    auto pv_polymorphic = make_PODVector<T, amrex::PolymorphicArenaAllocator<T>> (m, typestr, "polymorphic");

    // bind to_host/to_device now that every PODVector allocator class is
    // registered, so their PODVector return types resolve to known Python types
    add_host_device(
        pv_pinned,
        pv_arena,
        pv_std,
#ifdef AMREX_USE_GPU
        pv_device,
        pv_managed,
        pv_async,
#endif
        pv_polymorphic
    );

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
    nb::object const std_pod = m.attr(str_PODVector(typestr, "std").c_str());
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

void init_PODVector(nb::module_& m)
{
    nb::enum_<GrowthStrategy>(m, "GrowthStrategy")
        .value("Poisson", GrowthStrategy::Poisson)
        .value("Exact", GrowthStrategy::Exact)
        .value("Geometric", GrowthStrategy::Geometric)
        .export_values()
    ;

    make_PODVector<ParticleReal> (m, "real");
    make_PODVector<int> (m, "int");
    make_PODVector<uint64_t> (m, "uint64");
}
