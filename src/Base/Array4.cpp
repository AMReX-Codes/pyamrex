/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"


void init_Array4_float(nb::module_ &m);
void init_Array4_float_const(nb::module_ &m);

void init_Array4_complex(nb::module_ &m);
void init_Array4_complex_const(nb::module_ &m);

void init_Array4_int(nb::module_ &m);
void init_Array4_int_const(nb::module_ &m);

void init_Array4_uint(nb::module_ &m);
void init_Array4_uint_const(nb::module_ &m);

void init_Array4(nb::module_ &m)
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
    nb::class_< PolymorphicArray4, Array4 >(m, "PolymorphicArray4")
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
