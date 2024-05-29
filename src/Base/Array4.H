/* Copyright 2021-2023 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>

#include <complex>
#include <cstdint>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <vector>


namespace
{
    // helper type traits
    template <typename T>
    struct get_value_type { using value_type = T; };
    template <typename T>
    struct get_value_type<std::complex<T>> { using value_type = T; };
    template <typename T>
    using get_value_type_t = typename get_value_type<T>::value_type;

    // helper to check if Array4<T> is of constant value type T
    template <typename T>
    constexpr bool is_not_const ()
    {
        return std::is_same_v<
            std::remove_cv_t<
                T
            >,
            T
        > &&
        std::is_same_v<
            std::remove_cv_t<
                get_value_type_t<T>
            >,
            get_value_type_t<T>
        >;
    }
}

namespace pyAMReX
{
    using namespace amrex;

    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template<typename T>
    py::dict
    array_interface (Array4<T> const & a4)
    {
        auto d = py::dict();
        auto const len = length(a4);
        // F->C index conversion here
        // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
        // Buffer dimensions: zero-size shall not skip dimension
        auto shape = py::make_tuple(
            py::ssize_t(a4.ncomp),
            py::ssize_t(len.z <= 0 ? 1 : len.z),
            py::ssize_t(len.y <= 0 ? 1 : len.y),
            py::ssize_t(len.x <= 0 ? 1 : len.x)  // fastest varying index
        );
        // buffer protocol strides are in bytes, AMReX strides are elements
        auto const strides = py::make_tuple(
            py::ssize_t(sizeof(T) * a4.nstride),
            py::ssize_t(sizeof(T) * a4.kstride),
            py::ssize_t(sizeof(T) * a4.jstride),
            py::ssize_t(sizeof(T))  // fastest varying index
        );
        bool const read_only = false;  // note: we could decide on is_not_const,
                                       // but many libs, e.g. PyTorch, do not
                                       // support read-only and will raise
                                       // warnings, casting to read-write
        d["data"] = py::make_tuple(std::intptr_t(a4.dataPtr()), read_only);
        // note: if we want to keep the same global indexing with non-zero
        //       box small_end as in AMReX, then we can explore playing with
        //       this offset as well
        //d["offset"] = 0;         // default
        //d["mask"] = py::none();  // default

        d["shape"] = shape;
        // we could also set this after checking the strides are C-style contiguous:
        //if (is_contiguous<T>(shape, strides))
        //    d["strides"] = py::none();  // C-style contiguous
        //else
        d["strides"] = strides;

        // type description
        // for more complicated types, e.g., tuples/structs
        //d["descr"] = ...;
        // we currently only need this
        using T_no_cv = std::remove_cv_t<T>;
        d["typestr"] = py::format_descriptor<T_no_cv>::format();

        d["version"] = 3;
        return d;
    }

    template< typename T >
    void make_Array4(py::module &m, std::string typestr)
    {
        using namespace amrex;

        using T_no_cv = std::remove_cv_t<T>;

        // dispatch simpler via: py::format_descriptor<T>::format() naming
        // but note the _const suffix that might be needed
        auto const array_name = std::string("Array4_").append(typestr);
        py::class_< Array4<T> > py_array4(m, array_name.c_str());
        py_array4
            .def("__repr__",
                 [typestr](Array4<T> const & a4) {
                     std::stringstream s;
                     s << a4.size();
                     return "<amrex.Array4 of type '" + typestr +
                            "' and size '" + s.str() + "'>";
                 }
            )
    #if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
            .def("index_assert", &Array4<T>::index_assert)
    #endif

            .def_property_readonly("size", &Array4<T>::size)
            .def_property_readonly("nComp", &Array4<T>::nComp)
            .def_property_readonly("num_comp", &Array4<T>::nComp)

            .def(py::init< >())
            .def(py::init< Array4<T> const & >())
            .def(py::init< Array4<T> const &, int >())
            .def(py::init< Array4<T> const &, int, int >())
            //.def(py::init< T*, Dim3 const &, Dim3 const &, int >())

            /* init from a numpy or other buffer protocol array: non-owning view
             */
            .def(py::init([](py::array_t<T> & arr) {
                py::buffer_info buf = arr.request();

                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(buf.ndim == 3,
                    "We can only create amrex::Array4 views into 3D Python arrays at the moment.");
                // TODO:
                //   In 2D, Array4 still needs to be accessed with (i,j,k) or (i,j,k,n), with k = 0.
                //   Likewise in 1D.
                //   We could also add support for 4D numpy arrays, treating the slowest
                //   varying index as component "n".

                if (buf.format != py::format_descriptor<T_no_cv>::format())
                    throw std::runtime_error("Incompatible format: expected '" +
                        py::format_descriptor<T_no_cv>::format() +
                        "' and received '" + buf.format + "'!");

                auto a4 = std::make_unique< Array4<T> >();
                a4.get()->p = static_cast<T*>(buf.ptr);
                a4.get()->begin = Dim3{0, 0, 0};
                // C->F index conversion here
                // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
                a4.get()->end.x = (int)buf.shape.at(2); // fastest varying index
                a4.get()->end.y = (int)buf.shape.at(1);
                a4.get()->end.z = (int)buf.shape.at(0);
                a4.get()->ncomp = 1;
                // buffer protocol strides are in bytes, AMReX strides are elements
                a4.get()->jstride = (int)buf.strides.at(1) / sizeof(T); // fastest varying index
                a4.get()->kstride = (int)buf.strides.at(0) / sizeof(T);
                // 3D == no component: stride here should not matter
                a4.get()->nstride = a4.get()->kstride * (int)buf.shape.at(0);

                // todo: we could check and store here if the array buffer we got is read-only

                return a4;
            }))

            /* init from __cuda_array_interface__: non-owning view
             * TODO
             */


            // CPU: __array_interface__ v3
            // https://numpy.org/doc/stable/reference/arrays.interface.html
            .def_property_readonly("__array_interface__", [](Array4<T> const & a4) {
                return pyAMReX::array_interface(a4);
            })

            // CPU: __array_function__ interface (TODO)
            //
            // NEP 18 — A dispatch mechanism for NumPy's high level array functions.
            //   https://numpy.org/neps/nep-0018-array-function-protocol.html
            // This enables code using NumPy to be directly operated on Array4 arrays.
            // __array_function__ feature requires NumPy 1.16 or later.


            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            .def_property_readonly("__cuda_array_interface__", [](Array4<T> const & a4) {
                auto d = pyAMReX::array_interface(a4);

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


            // TODO: __dlpack__ __dlpack_device__
            // DLPack protocol (CPU, NVIDIA GPU, AMD GPU, Intel GPU, etc.)
            // https://dmlc.github.io/dlpack/latest/
            // https://data-apis.org/array-api/latest/design_topics/data_interchange.html
            // https://github.com/data-apis/consortium-feedback/issues/1
            // https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
            // https://docs.cupy.dev/en/stable/user_guide/interoperability.html#dlpack-data-exchange-protocol

            .def("to_host", [](Array4<T> const & a4) {
                // py::tuple to std::vector
                auto const a4i = pyAMReX::array_interface(a4);
                auto const shape = py::cast<std::vector<py::ssize_t>>(a4i["shape"]);
                auto const strides_bytes = py::cast<std::vector<py::ssize_t>>(a4i["strides"]);

                // allocate host memory copy
                auto h_data = py::array_t<T_no_cv>(shape, strides_bytes);

                // sync copy: host data is unpinned
                Gpu::copy(Gpu::deviceToHost,
                    a4.dataPtr(), a4.dataPtr() + a4.size(),
                    h_data.mutable_data()
                );
                return h_data;
            }, py::return_value_policy::move)

            //.def("__contains__", &Array4<T>::contains)  // syntax: "other in b"
            .def("contains", py::overload_cast<int, int, int>(&Array4<T>::contains, py::const_))
            .def("contains", py::overload_cast<IntVect const &>(&Array4<T>::contains, py::const_))
            .def("contains", py::overload_cast<Dim3 const &>(&Array4<T>::contains, py::const_))

            // getter
            .def("__getitem__", [](Array4<T> & a4, IntVect const & v){ return a4(v); })
            .def("__getitem__", [](Array4<T> & a4, std::array<int, 4> const key){
                return a4(key[0], key[1], key[2], key[3]);
            })
            .def("__getitem__", [](Array4<T> & a4, std::array<int, 3> const key){
                return a4(key[0], key[1], key[2]);
            })
        ;

        // setter
        if constexpr (is_not_const<T>())
        {
            py_array4
                .def("__setitem__", [](Array4<T> & a4, IntVect const & v, T const value){ a4(v) = value; })
                .def("__setitem__", [](Array4<T> & a4, std::array<int, 4> const key, T const value){
                    a4(key[0], key[1], key[2], key[3]) = value;
                })
                .def("__setitem__", [](Array4<T> & a4, std::array<int, 3> const key, T const value){
                    a4(key[0], key[1], key[2]) = value;
                })
            ;
        }

        // free standing C++ functions:
        m.def("lbound", &lbound< T >);
        m.def("ubound", &ubound< T >);
        m.def("length", &length< T >);
        //m.def("makePolymorphic", &makePolymorphic< T >);
    }
}
