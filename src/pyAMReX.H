/* Copyright 2021-2023 The AMReX Community
 *
 * This header is used to centrally define classes that shall not violate the
 * C++ one-definition-rule (ODR) for various Python translation units.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <AMReX_Config.H>
//include <AMReX.H>

#include <string>

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::list<...>)

namespace pyAMReX
{
    /** Array interface type string of T: byte order, kind and size, e.g., "<u8"
     *
     * NumPy resolves such a string to its canonical scalar type on any C data
     * model (LP32, ILP32, LLP64, LP64). In contrast, the PEP 3118 characters of
     * py::format_descriptor<T> are no type strings: they fail for complex numbers
     * and can select an alias of equal size that others do not handle, e.g.,
     * numpy.ulonglong instead of numpy.uint64 on LP64 or numpy.uintc instead of
     * numpy.uint32 on ILP32 and LLP64.
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <typename T>
    std::string array_typestr ()
    {
        return py::dtype::of<T>().attr("str").template cast<std::string>();
    }

    /** py::array_t<T> that also requires alignment
     *
     * A plain py::array_t<T> passes misaligned arrays through, e.g., from an offset
     * into a buffer or a field of a packed record, and accessing those through a T*
     * is undefined behavior. With this, NumPy makes an aligned copy instead.
     */
    template <typename T>
    using aligned_array_t = py::array_t<T,
        static_cast<int>(py::array::forcecast) | static_cast<int>(py::detail::npy_api::NPY_ARRAY_ALIGNED_)>;
}
