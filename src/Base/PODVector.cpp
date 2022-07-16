/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_PODVector.H>

#include <sstream>


namespace py = pybind11;
using namespace amrex;

template <class T, class Allocator = std::allocator<T> >
void make_PODVector(py::module &m, std::string typestr)
{
    using PODVector_type=PODVector<T, Allocator>;
    auto const podv_name = std::string("PODVector_").append(typestr);
    
    py::class_<PODVector_type>(m, podv_name.c_str())
        .def("__repr__",
             [typestr](PODVector_type const & pv) {
                 std::stringstream s, rs;
                 s << pv.size();
                 rs << "<amrex.PODVector of type '" + typestr +
                        "' and size '" + s.str() + "'>\n";
                 rs << "[ ";
                 for (int ii = 0; ii < int(pv.size()); ii++) {
                     rs << pv[ii] << " ";
                 }
                 rs << "]\n";
                 return rs.str();
             }
        )
        .def(py::init<>())
        .def(py::init<std::size_t>())
        .def(py::init<PODVector_type&>())
        .def("push_back", py::overload_cast<const T&>(&PODVector_type::push_back))
        .def("pop_back", &PODVector_type::pop_back)
        .def("clear", &PODVector_type::clear)
        .def("size", &PODVector_type::size)
        // .def("max_size", &PODVector_type::max_size)
        .def("capacity", &PODVector_type::capacity)
        .def("empty", &PODVector_type::empty)
        .def("resize", py::overload_cast<std::size_t>(&PODVector_type::resize))
        .def("resize", py::overload_cast<std::size_t, const T&>(&PODVector_type::resize))
        .def("reserve", &PODVector_type::reserve)
        .def("shrink_to_fit", &PODVector_type::shrink_to_fit)
        // TODO:
        // front
        // back
        // data
        // begin
        // 
        // swap

        .def_property_readonly("__array_interface__", [](PODVector_type const & podvector) {
            auto d = py::dict();
            bool const read_only = false;
            d["data"] = py::make_tuple(std::intptr_t(podvector.dataPtr()), read_only);
            d["shape"] = py::make_tuple(podvector.size());
            d["strides"] = py::none();
            d["typestr"] = py::format_descriptor<T>::format();
            d["version"] = 3;
            return d;
        })
        // setter & getter
        .def("__setitem__", [](PODVector_type & podvector, int const v, T const value){ podvector[v] = value; })
        .def("__getitem__", [](PODVector_type & pv, int const v){ return pv[v]; })
    ;
}

void init_PODVector(py::module& m) {
    make_PODVector<ParticleReal> (m, "real");
    make_PODVector<int> (m, "int");
}
