/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
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
        .def_property_readonly("num_comp", &Array4<T>::nComp)

        .def(py::init< >())
        .def(py::init< Array4<T> const & >())
        .def(py::init< Array4<T> const &, int >())
        .def(py::init< Array4<T> const &, int, int >())
        //.def(py::init< T*, Dim3 const &, Dim3 const &, int >())

        /* init from a numpy or other buffer protocol array: non-owning view
         */
        .def(py::init([](py::array_t<T> & arr) {
            py::buffer_info buf = arr.request();

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(buf.ndim == 3,
                "We can only create amrex::Array4 views into 3D Python arrays at the moment.");
            // TODO:
            //   In 2D, Array4 still needs to be accessed with (i,j,k) or (i,j,k,n), with k = 0.
            //   Likewise in 1D.
            //   We could also add support for 4D numpy arrays, treating the slowest
            //   varying index as component "n".

            if (buf.format != py::format_descriptor<T>::format())
                throw std::runtime_error("Incompatible format: expected '" +
                    py::format_descriptor<T>::format() +
                    "' and received '" + buf.format + "'!");

            auto a4 = std::make_unique< Array4<T> >();
            a4.get()->p = static_cast<T*>(buf.ptr);
            a4.get()->begin = Dim3{0, 0, 0};
            // C->F index conversion here
            // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
            a4.get()->end.x = (int)buf.shape.at(2); // fastest varying index
            a4.get()->end.y = (int)buf.shape.at(1);
            a4.get()->end.z = (int)buf.shape.at(0);
            a4.get()->ncomp = 1;
            // buffer protocol strides are in bytes, AMReX strides are elements
            a4.get()->jstride = (int)buf.strides.at(1) / sizeof(T); // fastest varying index
            a4.get()->kstride = (int)buf.strides.at(0) / sizeof(T);
            // 3D == no component: stride here should not matter
            a4.get()->nstride = a4.get()->kstride * (int)buf.shape.at(0);


            std::cout << "(int)buf.strides.at(0)=" << (int)buf.strides.at(0) << std::endl;
            std::cout << "(int)buf.strides.at(1)=" << (int)buf.strides.at(1) << std::endl;
            std::cout << "(int)buf.strides.at(2)=" << (int)buf.strides.at(2) << std::endl;

            // todo: we could check and store here if the array buffer we got is read-only

            return a4;
        }))

        .def_property_readonly("__array_interface__", [](Array4<T> const & a4) {
            auto d = py::dict();
            auto const len = length(a4);
            // F->C index conversion here
            // p[(i-begin.x)+(j-begin.y)*jstride+(k-begin.z)*kstride+n*nstride];
            //py::print("ncomp");
            //py::print(a4.ncomp);
            // Buffer dimensions: zero-size shall not have negative dimension
            auto shape = py::make_tuple(
                    a4.ncomp,
                    len.z < 0 ? 0 : len.z,
                    len.y < 0 ? 0 : len.y,
                    len.x < 0 ? 0 : len.x  // fastest varying index
            );
            // buffer protocol strides are in bytes, AMReX strides are elements
            auto const strides = py::make_tuple(  // Strides (in bytes) for each index
                    sizeof(T) * a4.nstride,
                    sizeof(T) * a4.kstride,
                    sizeof(T) * a4.jstride,
                    sizeof(T)  // fastest varying index
            );
            bool const read_only = false;
            d["data"] = py::make_tuple(long(a4.dataPtr()), read_only);
            //d["offset"] = 0;
            //d["mask"] = py::none();

            d["shape"] = shape;
            d["strides"] = strides;
            // we could set this after checking the strides are C-style contiguous:
            // d["strides"] = py::none();

            d["typestr"] = py::format_descriptor<T>::format();
            d["version"] = 3;
            return d;
        })


        // TODO: __cuda_array_interface__
        // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html


        // TODO: __dlpack__
        // DLPack protocol (CPU, NVIDIA GPU, AMD GPU, Intel GPU, etc.)
        // https://dmlc.github.io/dlpack/latest/
        // https://data-apis.org/array-api/latest/design_topics/data_interchange.html
        // https://github.com/data-apis/consortium-feedback/issues/1
        // https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h


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
        .def("contains", &Array4<T>::contains)
        //.def("__contains__", &Array4<T>::contains)

        // setter & getter
        .def("__setitem__", [](Array4<T> & a4, IntVect const & v, T const value){ a4(v) = value; })
        .def("__setitem__", [](Array4<T> & a4, std::array<int, 4> const key, T const value){
            a4(key[0], key[1], key[2], key[3]) = value;
        })
        .def("__setitem__", [](Array4<T> & a4, std::array<int, 3> const key, T const value){
            a4(key[0], key[1], key[2]) = value;
        })

        .def("__getitem__", [](Array4<T> & a4, IntVect const & v){ return a4(v); })
        .def("__getitem__", [](Array4<T> & a4, std::array<int, 4> const key){
            return a4(key[0], key[1], key[2], key[3]);
        })
        .def("__getitem__", [](Array4<T> & a4, std::array<int, 3> const key){
            return a4(key[0], key[1], key[2]);
        })
    ;

    // free standing C++ functions:
    m.def("lbound", &lbound< Array4<T> >);
    m.def("ubound", &ubound< Array4<T> >);
    m.def("length", &length< Array4<T> >);
    m.def("makePolymorphic", &makePolymorphic< Array4<T> >);
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
}
