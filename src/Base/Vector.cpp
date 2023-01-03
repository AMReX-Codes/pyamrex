/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_PODVector.H>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <optional>
#include <vector>

namespace py = pybind11;
using namespace amrex;


template <class T, class Allocator = std::allocator<T> >
void make_Vector(py::module &m, std::string typestr)
{
    using Vector_type=Vector<T, Allocator>;
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

        .def_property_readonly("__array_interface__", [](Vector_type const & vector) {
            auto d = py::dict();
            bool const read_only = false;
            d["data"] = py::make_tuple(std::intptr_t(vector.dataPtr()), read_only);
            d["shape"] = py::make_tuple(vector.size());
            d["strides"] = py::none();
            d["typestr"] = py::format_descriptor<T>::format();
            d["version"] = 3;
            return d;
        })
        // setter & getter
        .def("__setitem__", [](Vector_type & vector, int const idx, T const value){ vector[idx] = value; })
        .def("__getitem__", [](Vector_type & v, int const idx){ return v[idx]; })
    ;
}

void init_Vector(py::module& m) {
    make_Vector<Real> (m, "Real");
    if constexpr(!std::is_same_v<Real, ParticleReal>)
        make_Vector<ParticleReal> (m, "ParticleReal");

    make_Vector<int> (m, "int");
    if constexpr(!std::is_same_v<int, Long>)
        make_Vector<Long> (m, "Long");
}
