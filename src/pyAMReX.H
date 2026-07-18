/* Copyright 2021-2023 The AMReX Community
 *
 * This header is used to centrally define classes that shall not violate the
 * C++ one-definition-rule (ODR) for various Python translation units.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <AMReX_Config.H>
//include <AMReX.H>

#include <complex>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace pyAMReX
{
    /** PEP 3118 scalar format used by the existing array-interface contract. */
    template<typename T>
    std::string buffer_format ()
    {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, bool>) return "?";
        else if constexpr (std::is_same_v<U, signed char>) return "b";
        else if constexpr (std::is_same_v<U, unsigned char>) return "B";
        else if constexpr (std::is_same_v<U, short>) return "h";
        else if constexpr (std::is_same_v<U, unsigned short>) return "H";
        else if constexpr (std::is_same_v<U, int>) return "i";
        else if constexpr (std::is_same_v<U, unsigned int>) return "I";
        else if constexpr (std::is_same_v<U, long>) return "l";
        else if constexpr (std::is_same_v<U, unsigned long>) return "L";
        else if constexpr (std::is_same_v<U, long long>) return "q";
        else if constexpr (std::is_same_v<U, unsigned long long>) return "Q";
        else if constexpr (std::is_same_v<U, float>) return "f";
        else if constexpr (std::is_same_v<U, double>) return "d";
        else if constexpr (std::is_same_v<U, long double>) return "g";
        else if constexpr (std::is_same_v<U, std::complex<float>>) return "Zf";
        else if constexpr (std::is_same_v<U, std::complex<double>>) return "Zd";
        else static_assert(!sizeof(U), "Unsupported Python buffer scalar type");
    }
}
