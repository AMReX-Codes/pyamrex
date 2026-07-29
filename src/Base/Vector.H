/* Copyright 2022-2023 The AMReX Community
 *
 * Authors: Ryan Sandberg, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_Vector.H>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>


namespace
{
    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <class T, class Allocator = std::allocator<T> >
    nb::dict
    array_interface(amrex::Vector<T, Allocator> const & vector)
    {
        auto d = nb::dict();
        bool const read_only = false;
        d["data"] = nb::make_tuple(std::intptr_t(vector.dataPtr()), read_only);
        d["shape"] = nb::make_tuple(vector.size());
        d["strides"] = nb::none();
        d["typestr"] = pyAMReX::buffer_format<T>();
        d["version"] = 3;
        return d;
    }
}

/** Register an ``amrex::Vector`` type with nanobind.
 *
 * Specialized for a specific data type ``T`` and memory ``Allocator``.
 *
 * @tparam T data type per vector element
 * @tparam Allocator the memory allocator, e.g., an AMReX Arena or std::allocator
 * @param m the pyAMReX nanobind module object
 * @param typestr a unique suffix for the type ``T`` and ``Allocator``
 */
template <class T, class Allocator = std::allocator<T> >
void make_Vector(nb::module_ &m, std::string typestr)
{
    using namespace amrex;

    using Vector_type = Vector<T, Allocator>;
    auto const v_name = std::string("Vector_").append(typestr);

    auto py_vect = nb::bind_vector<Vector_type>(m, v_name.c_str());
    py_vect
        .def("__repr__",
             [typestr](Vector_type const & v) {
                 std::stringstream s, rs;
                 s << v.size();
                 rs << "<amrex.Vector of type '" + typestr +
                        "' and size '" + s.str() + "'>\n";
                 rs << "[ ";
                 for (int ii = 0; ii < int(v.size()); ii++) {
                     rs << v[ii] << " ";
                 }
                 rs << "]\n";
                 return rs.str();
             }
        )
        .def(nb::init<>())
        .def(nb::init<Vector_type const &>())

        .def("size", &Vector_type::size)
    ;

    if constexpr(std::is_fundamental_v<T>)
    {
      py_vect
        .def_prop_ro("__array_interface__", [](Vector_type const & vector) {
            return array_interface(vector);
        })
        .def_prop_ro("__cuda_array_interface__", [](Vector_type const & vector) {
            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            auto d = array_interface(vector);

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
        });
    }

    py_vect
        // setter & getter
        .def("__setitem__", [](Vector_type & vector, int const idx, T const value){ vector[idx] = value; })
        .def("__getitem__", [](Vector_type & v, int const idx){ return v[idx]; })
    ;
}
