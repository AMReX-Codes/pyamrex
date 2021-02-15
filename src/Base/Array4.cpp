/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Array4.H>
#include <AMReX_IntVect.H>

#include <sstream>
#include <type_traits>

namespace py = pybind11;
using namespace amrex;


template< typename T >
void make_Array4(py::module &m, std::string typestr)
{
    // dispatch simpler via: py::format_descriptor<T>::format() naming
    auto const array_name = std::string("Array4_").append(typestr);
    py::class_< Array4<T> >(m, array_name.c_str(), py::buffer_protocol())
        .def("__repr__",
             [](Array4<T> const & a4) {
                 std::stringstream s;
                 s << a4.size();
                 return "<amrex.Array4 of size '" + s.str() + "'>";
             }
        )
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        .def("index_assert", &Array4<T>::index_assert)
#endif

        .def_property_readonly("size", &Array4<T>::size)
        .def_property_readonly("nComp", &Array4<T>::nComp)

        .def(py::init< >())
        .def(py::init< Array4<T> const & >())
        .def(py::init< Array4<T> const &, int >())
        .def(py::init< Array4<T> const &, int, int >())
        //.def(py::init< T*, Dim3 const &, Dim3 const &, int >())

        .def(py::init([](py::array_t<T> & arr) {
            py::buffer_info buf = arr.request();

            auto a4 = std::make_unique< Array4<T> >();
            a4.get()->p = (T*)buf.ptr;
            a4.get()->begin = Dim3{0, 0, 0};
            // TODO: likely C->F index conversion here
            // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
            a4.get()->end.x = (int)buf.shape.at(0);
            a4.get()->end.y = (int)buf.shape.at(1);
            a4.get()->end.z = (int)buf.shape.at(2);
            a4.get()->ncomp = 1;
            // buffer protocol strides are in bytes, AMReX strides are elements
            a4.get()->jstride = (int)buf.strides.at(0) / sizeof(T);
            a4.get()->kstride = (int)buf.strides.at(1) / sizeof(T);
            a4.get()->nstride = (int)buf.strides.at(2) * (int)buf.shape.at(2) / sizeof(T);
            return a4;
        }))


        .def_property_readonly("__array_interface__", [](Array4<T> const & a4) {
            auto d = py::dict();
            auto const len = length(a4);
            // TODO: likely F->C index conversion here
            // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
            auto shape = py::make_tuple(  // Buffer dimensions
                    len.x < 0 ? 0 : len.x,
                    len.y < 0 ? 0 : len.y,
                    len.z < 0 ? 0 : len.z//,  // zero-size shall not have negative dimension
                    //a4.ncomp
            );
            // buffer protocol strides are in bytes, AMReX strides are elements
            auto const strides = py::make_tuple(  // Strides (in bytes) for each index
                    sizeof(T) * a4.jstride,
                    sizeof(T) * a4.kstride,
                    sizeof(T)//,
                    //sizeof(T) * a4.nstride
            );
            d["data"] = py::make_tuple(long(a4.dataPtr()), false);
            d["typestr"] = py::format_descriptor<T>::format();
            d["shape"] = shape;
            d["strides"] = strides;
            // d["strides"] = py::none();
            d["version"] = 3;
            return d;
        })

        // not sure if useful to have this implemented on top
/*
        .def_buffer([](Array4<T> & a4) -> py::buffer_info {
            auto const len = length(a4);
            // TODO: likely F->C index conversion here
            // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
            auto shape = {  // Buffer dimensions
                    len.x < 0 ? 0 : len.x,
                    len.y < 0 ? 0 : len.y,
                    len.z < 0 ? 0 : len.z//,  // zero-size shall not have negative dimension
                    //a4.ncomp
            };
            // buffer protocol strides are in bytes, AMReX strides are elements
            auto const strides = {  // Strides (in bytes) for each index
                    sizeof(T) * a4.jstride,
                    sizeof(T) * a4.kstride,
                    sizeof(T)//,
                    //sizeof(T) * a4.nstride
            };
            return py::buffer_info(
                a4.dataPtr(),
                shape,
                strides
            );
        })
*/
    ;
}

void init_Array4(py::module &m) {
    make_Array4< float >(m, "float");
    make_Array4< double >(m, "double");
    make_Array4< long double >(m, "longdouble");

    make_Array4< short >(m, "short");
    make_Array4< int >(m, "int");
    make_Array4< long >(m, "long");
    make_Array4< long long >(m, "longlong");

    make_Array4< unsigned short >(m, "ushort");
    make_Array4< unsigned int >(m, "uint");
    make_Array4< unsigned long >(m, "ulong");
    make_Array4< unsigned long long >(m, "ulonglong");

    // std::complex< float|double|long double> ?

    /*
    py::class_< PolymorphicArray4, Array4 >(m, "PolymorphicArray4")
        .def("__repr__",
             [](PolymorphicArray4 const & pa4) {
                 std::stringstream s;
                 s << pa4.size();
                 return "<amrex.PolymorphicArray4 of size '" + s.str() + "'>";
             }
        )
    ;
     */

    // free standing C++ functions:
    /*
    contains
    lbound
    ubound
    length
    makePolymorphic
    */
}
