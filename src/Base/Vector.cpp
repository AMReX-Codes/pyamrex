/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_PODVector.H>

#include <sstream>
#include <string>
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

        /* init from a numpy or other buffer protocol array: non-owning view
         */
        // .def(py::init([](py::array_t<T> & arr) {
        //     py::buffer_info buf = arr.request();

        //     if (buf.format != py::format_descriptor<T>::format())
        //         throw std::runtime_error("Incompatible format: expected '" +
        //             py::format_descriptor<T>::format() +
        //             "' and received '" + buf.format + "'!");

        //     auto v = std::make_unique< Vector<T> >();
        //     v.get()->data() = static_cast<T*>(buf.ptr);
        //     return v;
        // }))
        // .def(py::init<std::size_t>())
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
    make_Vector<ParticleReal> (m, "real");
    make_Vector<int> (m, "int");
    make_Vector<Long> (m, "Long");
    // make_Vector<std::string> (m, "string");
}