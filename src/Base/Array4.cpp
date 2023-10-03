/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"


void init_Array4_float(py::module &m);
void init_Array4_float_const(py::module &m);

void init_Array4_complex(py::module &m);
void init_Array4_complex_const(py::module &m);

void init_Array4_int(py::module &m);
void init_Array4_int_const(py::module &m);

void init_Array4_uint(py::module &m);
void init_Array4_uint_const(py::module &m);

void init_Array4(py::module &m)
{
    using namespace pyAMReX;

    init_Array4_float(m);
    init_Array4_float_const(m);

    init_Array4_complex(m);
    init_Array4_complex_const(m);

    init_Array4_int(m);
    init_Array4_int_const(m);

    init_Array4_uint(m);
    init_Array4_uint_const(m);

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
