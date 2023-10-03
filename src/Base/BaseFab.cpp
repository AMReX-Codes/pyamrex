/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Array4.H"

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

            .def("array", [](BaseFab<T> & bf)
                { return bf.array(); },
                // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
                py::keep_alive<0, 1>()
            )
            .def("const_array", [](BaseFab<T> const & bf)
                { return bf.const_array(); },
                // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
                 py::keep_alive<0, 1>()
            )

            .def("to_host", [](BaseFab<T> const & bf) {
                BaseFab<T> hbf(bf.box(), bf.nComp(), The_Pinned_Arena());
                Array4<T> ha = hbf.array();
                Gpu::copyAsync(Gpu::deviceToHost,
                    bf.dataPtr(), bf.dataPtr() + bf.size(),
                    ha.dataPtr());
                Gpu::streamSynchronize();
                return hbf;
            }, py::return_value_policy::move)

            // CPU: __array_interface__ v3
            // https://numpy.org/doc/stable/reference/arrays.interface.html
            .def_property_readonly("__array_interface__", [](BaseFab<T> const & bf) {
                return pyAMReX::array_interface(bf.array());
            })

            // CPU: __array_function__ interface (TODO)
            //
            // NEP 18 — A dispatch mechanism for NumPy's high level array functions.
            //   https://numpy.org/neps/nep-0018-array-function-protocol.html
            // This enables code using NumPy to be directly operated on Array4 arrays.
            // __array_function__ feature requires NumPy 1.16 or later.


            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            .def_property_readonly("__cuda_array_interface__", [](BaseFab<T> & bf) {
                auto d = pyAMReX::array_interface(bf.array());

                // data:
                // Because the user of the interface may or may not be in the same context, the most common case is to use cuPointerGetAttribute with CU_POINTER_ATTRIBUTE_DEVICE_POINTER in the CUDA driver API (or the equivalent CUDA Runtime API) to retrieve a device pointer that is usable in the currently active context.
                // TODO For zero-size arrays, use 0 here.

                // None or integer
                // An optional stream upon which synchronization must take place at the point of consumption, either by synchronizing on the stream or enqueuing operations on the data on the given stream. Integer values in this entry are as follows:
                //   0: This is disallowed as it would be ambiguous between None and the default stream, and also between the legacy and per-thread default streams. Any use case where 0 might be given should either use None, 1, or 2 instead for clarity.
                //   1: The legacy default stream.
                //   2: The per-thread default stream.
                //   Any other integer: a cudaStream_t represented as a Python integer.
                //   When None, no synchronization is required.
                d["stream"] = py::none();

                d["version"] = 3;
                return d;
            })


            // TODO: __dlpack__ __dlpack_device__
            // DLPack protocol (CPU, NVIDIA GPU, AMD GPU, Intel GPU, etc.)
            // https://dmlc.github.io/dlpack/latest/
            // https://data-apis.org/array-api/latest/design_topics/data_interchange.html
            // https://github.com/data-apis/consortium-feedback/issues/1
            // https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
            // https://docs.cupy.dev/en/stable/user_guide/interoperability.html#dlpack-data-exchange-protocol

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
