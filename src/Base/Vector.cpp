/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_PODVector.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/numpy.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <optional>
#include <vector>

namespace py = nanobind;
using namespace amrex;

namespace
{
    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <class T, class Allocator = std::allocator<T> >
    py::dict
    array_interface(Vector<T, Allocator> const & vector)
    {
        auto d = py::dict();
        bool const read_only = false;
        d["data"] = py::make_tuple(std::intptr_t(vector.dataPtr()), read_only);
        d["shape"] = py::make_tuple(vector.size());
        d["strides"] = py::none();
        d["typestr"] = py::format_descriptor<T>::format();
        d["version"] = 3;
        return d;
    }
}

template <class T, class Allocator = std::allocator<T> >
void make_Vector(py::module_ &m, std::string typestr)
{
    using Vector_type = Vector<T, Allocator>;
    auto const v_name = std::string("Vector_").append(typestr);

    py::class_<Vector_type>(m, v_name.c_str())
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
        .def(py::init<>())

        .def("size", &Vector_type::size)

        .def_prop_rw_readonly("__array_interface__", [](Vector_type const & vector) {
            return array_interface(vector);
        })
        .def_prop_rw_readonly("__cuda_array_interface__", [](Vector_type const & vector) {
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
            d["stream"] = py::none();

            d["version"] = 3;
            return d;
        })

        // setter & getter
        .def("__setitem__", [](Vector_type & vector, int const idx, T const value){ vector[idx] = value; })
        .def("__getitem__", [](Vector_type & v, int const idx){ return v[idx]; })
    ;
}

void init_Vector(py::module_& m) {
    make_Vector<Real> (m, "Real");
    if constexpr(!std::is_same_v<Real, ParticleReal>)
        make_Vector<ParticleReal> (m, "ParticleReal");

    make_Vector<int> (m, "int");
    if constexpr(!std::is_same_v<int, Long>)
        make_Vector<Long> (m, "Long");
}
