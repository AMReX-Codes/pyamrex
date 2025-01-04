/* Copyright 2021-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_SmallMatrix.H>

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
    template<
        class T,
        int NRows,
        int NCols,
        Order ORDER = Order::F,
        int StartIndex = 0
    >
    py::dict
    array_interface (SmallMatrix<T, NRows, NCols, ORDER, StartIndex> const & m)
    {
        auto d = py::dict();
        // provide C index order for shape and strides
        auto shape = m.ordering == Order::F ? py::make_tuple(
            py::ssize_t(NRows),
            py::ssize_t(NCols)  // fastest varying index
        ) : py::make_tuple(
            py::ssize_t(NCols),
            py::ssize_t(NRows)  // fastest varying index
        );
        // buffer protocol strides are in bytes
        auto const strides = m.ordering == Order::F ? py::make_tuple(
            py::ssize_t(sizeof(T) * NCols),
            py::ssize_t(sizeof(T))  // fastest varying index
        ) : py::make_tuple(
            py::ssize_t(sizeof(T) * NRows),
            py::ssize_t(sizeof(T))  // fastest varying index
        );
        bool const read_only = false;  // note: we could decide on is_not_const,
                                       // but many libs, e.g. PyTorch, do not
                                       // support read-only and will raise
                                       // warnings, casting to read-write
        d["data"] = py::make_tuple(std::intptr_t(&m.template get<0>()), read_only);
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

    template<
        class T,
        int NRows,
        int NCols,
        Order ORDER = Order::F,
        int StartIndex = 0
    >
    void make_SmallMatrix(py::module &m, std::string typestr)
    {
        using namespace amrex;

        using T_no_cv = std::remove_cv_t<T>;

        // dispatch simpler via: py::format_descriptor<T>::format() naming
        // but note the _const suffix that might be needed
        auto const sm_name = std::string("SmallMatrix_")
            .append(std::to_string(NRows)).append("x").append(std::to_string(NCols))
            .append("_").append(ORDER == Order::F ? "F" : "C")
            .append("_SI").append(std::to_string(StartIndex))
            .append("_").append(typestr);
        using SM = SmallMatrix<T, NRows, NCols, ORDER, StartIndex>;
        py::class_< SM > py_sm(m, sm_name.c_str());
        py_sm
            .def("__repr__",
                 [sm_name](SM const &) {
                     return "<amrex." + sm_name + ">";
                 }
            )
            .def("__str__",
                 [sm_name](SM const & sm) {
                     std::stringstream ss;
                     ss << sm;
                     return ss.str();
                 }
            )

            .def_property_readonly("size", [](SM const &){ return SM::row_size * SM::column_size; })
            .def_property_readonly("row_size", [](SM const &){ return SM::row_size; })
            .def_property_readonly("column_size", [](SM const &){ return SM::column_size; })
            .def_property_readonly("order", [](SM const &){ return SM::ordering == Order::F ? "F" : "C"; })  // NumPy name
            .def_property_readonly("starting_index", [](SM const &){ return SM::starting_index; })

            .def_static("zero", [](){ return SM::Zero(); })

            .def(py::init([](){ return SM{}; }))  // zero-init
            .def(py::init<SM const &>())          // copy-init

            /* init from a numpy or other buffer protocol array: copy
             */
            .def(py::init([](py::array_t<T> & arr)
            {
                py::buffer_info buf = arr.request();

                constexpr bool is_vector = SM::column_size == 1 || SM::row_size == 1;
                constexpr int sm_dim = is_vector ? 1 : 2;
                if (buf.ndim != sm_dim)
                    throw std::runtime_error("The SmallMatrix to create is " + std::to_string(sm_dim) +
                                             "D, but the passed array is " + std::to_string(buf.ndim) + "D.");
                if (buf.size != SM::column_size * SM::row_size)
                    throw std::runtime_error("Array size mismatch: Expected " + std::to_string(SM::column_size * SM::row_size) +
                                             " elements, but passed " + std::to_string(buf.size) + " elements.");

                if (buf.format != py::format_descriptor<T_no_cv>::format())
                    throw std::runtime_error("Incompatible format: expected '" +
                        py::format_descriptor<T_no_cv>::format() +
                        "' and received '" + buf.format + "'!");

                // TODO: check that strides are either exact or None in buf
                // TODO: transpose if SM order is not C?

                auto sm = std::make_unique< SM >();
                auto * src = static_cast<T*>(buf.ptr);
                std::copy(src, src + buf.size, &sm->template get<0>());

                // todo: we could check and store here if the array buffer we got is read-only

                return sm;
            }))

            /* init from __cuda_array_interface__: device-to-host copy
             * TODO
             */


            // CPU: __array_interface__ v3
            // https://numpy.org/doc/stable/reference/arrays.interface.html
            .def_property_readonly("__array_interface__", [](SM const & sm) {
                return pyAMReX::array_interface(sm);
            })

            // CPU: __array_function__ interface (TODO)
            //
            // NEP 18 — A dispatch mechanism for NumPy's high level array functions.
            //   https://numpy.org/neps/nep-0018-array-function-protocol.html
            // This enables code using NumPy to be directly operated on Array4 arrays.
            // __array_function__ feature requires NumPy 1.16 or later.


            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            .def_property_readonly("__cuda_array_interface__", [](SM const & sm) {
                auto d = pyAMReX::array_interface(sm);

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

            .def("dot", &SM::dot)
            .def("prod", &SM::product)  // NumPy name
            .def("set_val", &SM::setVal)
            .def("sum", &SM::sum)
            .def_property_readonly("T", &SM::transpose)   // NumPy name

            // operators
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * amrex::Real())
            .def(amrex::Real() * py::self)
            .def(-py::self)

            // getter
            .def("__getitem__", [](SM & sm, std::array<int, 2> const & key){
                if (key[0] < SM::starting_index || key[0] >= SM::row_size + SM::starting_index ||
                    key[1] < SM::starting_index || key[1] >= SM::column_size + SM::starting_index)
                    throw std::runtime_error(
                        "Index out of bounds: [" +
                        std::to_string(key[0]) + ", " +
                        std::to_string(key[1]) + "]");
                return sm(key[0], key[1]);
            })
        ;
        // setter
        if constexpr (is_not_const<T>())
        {
            py_sm
                .def("__setitem__", [](SM & sm, std::array<int, 2> const & key, T const value){
                    if (key[0] < SM::starting_index || key[0] >= SM::row_size + SM::starting_index ||
                        key[1] < SM::starting_index || key[1] >= SM::column_size + SM::starting_index)
                        throw std::runtime_error(
                            "Index out of bounds: [" +
                            std::to_string(key[0]) + ", " +
                            std::to_string(key[1]) + "]");
                    sm(key[0], key[1]) = value;
                })
            ;
        }

        // square matrix
        if constexpr (NRows == NCols)
        {
            py_sm
                .def_static("identity", [](){ return SM::Identity(); })
                .def("trace", &SM::trace)
                .def("transpose_in_place", &SM::transposeInPlace)
            ;
        }

        // vector
        if constexpr (NRows == 1 || NCols == 1)
        {
            py_sm
                .def("__getitem__", [](SM & sm, int key){
                    if (key < SM::starting_index || key >= SM::column_size * SM::row_size + SM::starting_index)
                        throw std::runtime_error("Index out of bounds: " + std::to_string(key));
                    return sm(key);
                })
                .def("__setitem__", [](SM & sm, int key, T const value){
                    if (key < SM::starting_index || key >= SM::column_size * SM::row_size + SM::starting_index)
                        throw std::runtime_error("Index out of bounds: " + std::to_string(key));
                    sm(key) = value;
                })
            ;
        } else {
            using SV = SmallMatrix<T, NRows, 1, Order::F, StartIndex>;
            using SRV = SmallMatrix<T, 1, NCols, Order::F, StartIndex>;

            // operators for matrix-matrix & matrix-vector
            py_sm
                .def(py::self * py::self)
                .def(py::self * SV())
                .def(SRV() * py::self)
            ;
        }
    }
}
