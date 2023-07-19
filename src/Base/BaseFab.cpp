/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_FArrayBox.H>

#include <istream>


namespace
{
    using namespace amrex;

    template< typename T >
    void init_bf(py::module &m, std::string typestr) {
        auto const bf_name = std::string("BaseFab_").append(typestr);
        py::class_< BaseFab<T> >(m, bf_name.c_str())
            .def("__repr__",
                 [bf_name](BaseFab<Real> const & bf) {
                     std::string r = "<amrex.";
                     r.append(bf_name).append(" (n_comp=");
                     r.append(std::to_string(bf.nComp())).append(")>");
                     return r;
                 }
            )

            .def(py::init< >())
            .def(py::init< Arena* >())
            .def(py::init< Box const &, int, Arena* >())
            //.def(py::init< >( Box const &, int, bool, bool, Arena* ))
            //.def(py::init< const BaseFab<T>&, MakeType, int, int >())
            // non-owning
            .def(py::init< const Box&, int, T* >())
            .def(py::init< const Box&, int, T const* >())

            .def(py::init< Array4<T> const& >())
            .def(py::init< Array4<T> const&, IndexType >())
            .def(py::init< Array4<T const> const& >())
            .def(py::init< Array4<T const> const&, IndexType >())

            //.def_static("initialize", &BaseFab<T>::Initialize )
            //.def_static("finalize", &BaseFab<T>::Finalize )

            .def("resize", &BaseFab<T>::resize )
            .def("clear", &BaseFab<T>::clear )
            //.def("release", &BaseFab<T>::release )

            .def("n_bytes", py::overload_cast< >(&BaseFab<T>::nBytes, py::const_))
            .def("n_bytes", py::overload_cast< Box const &, int >(&BaseFab<T>::nBytes, py::const_))
            .def("n_bytes_owned", &BaseFab<T>::nBytesOwned )
            .def("n_comp", &BaseFab<T>::nComp )
            .def("num_pts", &BaseFab<T>::numPts )
            .def("size", &BaseFab<T>::size )
            .def("box", &BaseFab<T>::box )
            .def("length", &BaseFab<T>::length )

            .def("small_end", &BaseFab<T>::smallEnd )
            .def("big_end", &BaseFab<T>::bigEnd )
            .def("lo_vect", &BaseFab<T>::loVect )
            .def("hi_vect", &BaseFab<T>::hiVect )

            // contains
            // prefetchToHost
            // prefetchToDevice
            .def("is_allocated", &BaseFab<T>::isAllocated )

            //.def("array", &BaseFab<T>::array )
            //.def("const_array", &BaseFab<T>::const_array )

            // getVal
            // setVal
            // setValIf
            // setValIfNot
            // setComplement
            // copy
            // copyToMem
            // copyFromMem
            // addFromMem

            // shift
            // shiftHalf

            // norminfmask
            // norm
            // abs
            // min
            // max
            // minmax
            // maxabs
            // indexFromValue
            // minIndex
            // maxIndex
            // maskLT
            // maskLE
            // maskEQ
            // maskGT
            // maskGE
            // sum
            // invert
            // negate
            // plus
            // atomicAdd
            // saxpy
            // xpay
            // addproduct
            // minus
            // mult
            // divide
            // protected_divide
            // linInterp
            // linComb
            // dot
            // dotmask

            // SetBoxType
        ;
    }
}

void init_BaseFab(py::module &m) {
    using namespace amrex;

    init_bf<Real>(m, "Real");
}
