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
        // .def(py::init([](py::array_t<T> & arr) {
        //     py::buffer_info buf = arr.request();

        //     AMREX_ALWAYS_ASSERT_WITH_MESSAGE(buf.ndim == 3,
        //         "We can only create amrex::Array4 views into 3D Python arrays at the moment.");
        //     // TODO:
        //     //   In 2D, Array4 still needs to be accessed with (i,j,k) or (i,j,k,n), with k = 0.
        //     //   Likewise in 1D.
        //     //   We could also add support for 4D numpy arrays, treating the slowest
        //     //   varying index as component "n".

        //     if (buf.format != py::format_descriptor<T>::format())
        //         throw std::runtime_error("Incompatible format: expected '" +
        //             py::format_descriptor<T>::format() +
        //             "' and received '" + buf.format + "'!");

        //     auto pv = std::make_unique< PODVector_type >();
        //     pv.get()->data() = static_cast<T*>(buf.ptr);
        //     pv.
        //     // pv.get()->begin = Dim3{0, 0, 0};
        //     // C->F index conversion here
        //     // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
        //     // pv.get()->end.x = (int)buf.shape.at(2); // fastest varying index
        //     // pv.get()->end.y = (int)buf.shape.at(1);
        //     // pv.get()->end.z = (int)buf.shape.at(0);
        //     // pv.get()->ncomp = 1;
        //     // buffer protocol strides are in bytes, AMReX strides are elements
        //     // pv.get()->jstride = (int)buf.strides.at(1) / sizeof(T); // fastest varying index
        //     // pv.get()->kstride = (int)buf.strides.at(0) / sizeof(T);
        //     // 3D == no component: stride here should not matter
        //     // pv.get()->nstride = .get()->kstride * (int)buf.shape.at(0);

        //     // todo: we could check and store here if the array buffer we got is read-only

        //     return pv;
        // }))
        // .def(py::init<std::initializer_list<T>, const )
        // .def erase
        // insert
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
        // []
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
