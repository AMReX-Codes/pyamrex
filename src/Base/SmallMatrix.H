/* Copyright 2021-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_SmallMatrix.H>

#include <complex>
#include <cstdint>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <vector>


namespace
{
    // helper type traits
    template <typename T>
    struct get_value_type { using value_type = T; };
    template <typename T>
    struct get_value_type<std::complex<T>> { using value_type = T; };
    template <typename T>
    using get_value_type_t = typename get_value_type<T>::value_type;

    // helper to check if Array4<T> is of constant value type T
    template <typename T>
    constexpr bool is_not_const ()
    {
        return std::is_same_v<
            std::remove_cv_t<
                T
            >,
            T
        > &&
        std::is_same_v<
            std::remove_cv_t<
                get_value_type_t<T>
            >,
            get_value_type_t<T>
        >;
    }

    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template<
        class T,
        int NRows,
        int NCols,
        amrex::Order ORDER = amrex::Order::F,
        int StartIndex = 0
    >
    nb::dict
    array_interface (amrex::SmallMatrix<T, NRows, NCols, ORDER, StartIndex> const & m)
    {
        using namespace amrex;

        auto d = nb::dict();
        // provide C index order for shape and strides
        auto shape = m.ordering == Order::F ? nb::make_tuple(
            nb::ssize_t(NRows),
            nb::ssize_t(NCols)  // fastest varying index
        ) : nb::make_tuple(
            nb::ssize_t(NCols),
            nb::ssize_t(NRows)  // fastest varying index
        );
        // buffer protocol strides are in bytes
        auto const strides = m.ordering == Order::F ? nb::make_tuple(
            nb::ssize_t(sizeof(T) * NCols),
            nb::ssize_t(sizeof(T))  // fastest varying index
        ) : nb::make_tuple(
            nb::ssize_t(sizeof(T) * NRows),
            nb::ssize_t(sizeof(T))  // fastest varying index
        );
        bool const read_only = false;  // note: we could decide on is_not_const,
                                       // but many libs, e.g. PyTorch, do not
                                       // support read-only and will raise
                                       // warnings, casting to read-write
        d["data"] = nb::make_tuple(std::intptr_t(&m.template get<0>()), read_only);
        // note: if we want to keep the same global indexing with non-zero
        //       box small_end as in AMReX, then we can explore playing with
        //       this offset as well
        //d["offset"] = 0;         // default
        //d["mask"] = nb::none();  // default

        d["shape"] = shape;
        // we could also set this after checking the strides are C-style contiguous:
        //if (is_contiguous<T>(shape, strides))
        //    d["strides"] = nb::none();  // C-style contiguous
        //else
        d["strides"] = strides;

        // type description
        // for more complicated types, e.g., tuples/structs
        //d["descr"] = ...;
        // we currently only need this
        using T_no_cv = std::remove_cv_t<T>;
        d["typestr"] = pyAMReX::buffer_format<T_no_cv>();

        d["version"] = 3;
        return d;
    }

    template<class SM>
    void add_matrix_methods (nb::class_<SM> & py_sm)
    {
        using T = typename SM::value_type;
        using T_no_cv = std::remove_cv_t<T>;
        static constexpr int row_size = SM::row_size;
        static constexpr int column_size = SM::column_size;
        static constexpr int starting_index = SM::starting_index;

        py_sm
            .def("dot", &SM::dot)
            .def("prod", &SM::product)  // NumPy name
            .def("set_val", &SM::setVal)
            .def("sum", &SM::sum)
            .def_prop_ro("T", &SM::transpose)   // NumPy name

            // operators
            .def(nb::self + nb::self)
            .def(nb::self - nb::self)
            .def(nb::self * amrex::Real())
            .def(amrex::Real() * nb::self)
            .def(-nb::self)

            // getter
            .def("__getitem__", [](SM & sm, std::array<int, 2> const & key){
                if (key[0] < starting_index || key[0] >= row_size + starting_index ||
                    key[1] < starting_index || key[1] >= column_size + starting_index)
                    throw std::runtime_error(
                        "Index out of bounds: [" +
                        std::to_string(key[0]) + ", " +
                        std::to_string(key[1]) + "]");
                return sm(key[0], key[1]);
            })
        ;

        // setter
        if constexpr (is_not_const<T>())
        {
            py_sm
                .def("__setitem__", [](SM & sm, std::array<int, 2> const & key, T_no_cv const value){
                    if (key[0] < SM::starting_index || key[0] >= SM::row_size + SM::starting_index ||
                        key[1] < SM::starting_index || key[1] >= SM::column_size + SM::starting_index)
                    {
                        throw std::runtime_error(
                            "Index out of bounds: [" +
                            std::to_string(key[0]) + ", " +
                            std::to_string(key[1]) + "]");
                    }
                    sm(key[0], key[1]) = value;
                })
            ;
        }

        // square matrix
        if constexpr (row_size == column_size)
        {
            py_sm
                .def_static("identity", []() { return SM::Identity(); })
                .def("trace", [](SM & sm){ return sm.trace(); })
                .def("transpose_in_place", [](SM & sm){ return sm.transposeInPlace(); })
            ;
        }
    }

    template<class T_SV>
    void add_get_set_Vector (nb::class_<T_SV> &py_v)
    {
        using self = T_SV;
        using T = typename T_SV::value_type;
        using T_no_cv = std::remove_cv_t<T>;

        py_v
            .def("__getitem__", [](self & sm, int key){
                if (key < self::starting_index || key >= self::column_size * self::row_size + self::starting_index)
                    throw std::runtime_error("Index out of bounds: " + std::to_string(key));
                return sm(key);
            })
            .def("__setitem__", [](self & sm, int key, T_no_cv const value){
                if (key < self::starting_index || key >= self::column_size * self::row_size + self::starting_index)
                    throw std::runtime_error("Index out of bounds: " + std::to_string(key));
                sm(key) = value;
            })
        ;
    }

    template<class SM>
    nb::class_<SM>
    make_SmallMatrix_or_Vector (nb::module_ &m, std::string typestr)
    {
        using namespace amrex;

        // A vector specialization can be shared by several matrix bindings.
        // Use nanobind's documented public type registry to reuse it.
        if (nb::handle registered = nb::type<SM>(); registered.is_valid())
            return nb::borrow<nb::class_<SM>>(registered);

        using T = typename SM::value_type;
        static constexpr int row_size = SM::row_size;
        static constexpr int column_size = SM::column_size;
        static constexpr Order ordering = SM::ordering;
        static constexpr int starting_index = SM::starting_index;

        // dispatch simpler via: pyAMReX::buffer_format<T>() naming
        // but note the _const suffix that might be needed
        auto const sm_name = std::string("SmallMatrix_")
            .append(std::to_string(row_size)).append("x").append(std::to_string(column_size))
            .append("_").append(ordering == Order::F ? "F" : "C")
            .append("_SI").append(std::to_string(starting_index))
            .append("_").append(typestr);
        nb::class_< SM > py_sm(m, sm_name.c_str());
        py_sm
            .def("__repr__",
                [](SM const & sm) {
                     // similar look & feel like numpy.ndarray.__repr__
                     std::stringstream ss;
                     for (int i = sm.starting_index; i < sm.row_size + sm.starting_index; ++i) {
                         if (i > sm.starting_index) { ss << ",\n             "; }
                         ss << "[";
                         ss << sm(i, sm.starting_index);
                         for (int j = 1 + sm.starting_index; j < sm.column_size + sm.starting_index; ++j) {
                            ss << ", " << sm(i, j);
                         }
                         ss << "]";
                     }
                     using namespace std::string_literals;
                     return "SmallMatrix(["s +
                            ss.str() +
                            "])"s;
                 }
            )

            .def_prop_ro("size", [](SM const &){ return SM::row_size * SM::column_size; })
            .def_prop_ro("row_size", [](SM const &){ return SM::row_size; })
            .def_prop_ro("column_size", [](SM const &){ return SM::column_size; })
            .def_prop_ro("order", [](SM const &){ return SM::ordering == Order::F ? "F" : "C"; })  // NumPy name
            .def_prop_ro("starting_index", [](SM const &){ return SM::starting_index; })

            .def_static("zero", [](){ return SM::Zero(); })

            .def("__init__", [](SM * self) { new (self) SM{}; })  // zero-init
            .def(nb::init<SM const &>())          // copy-init

            /* init from a numpy or other buffer protocol array: copy
             */
            .def("__init__", [](SM * self, nb::object const& input)
            {
                auto const dtype = pyAMReX::buffer_format<T>();
                auto normalized = nb::module_::import_("numpy").attr("asarray")(
                    input,
                    nb::arg("dtype") = nb::str(dtype.c_str()),
                    nb::arg("order") = "C");
                auto arr = nb::cast<nb::ndarray<nb::numpy, T>>(normalized);
                constexpr bool is_vector = SM::column_size == 1 || SM::row_size == 1;
                constexpr int sm_dim = is_vector ? 1 : 2;
                if (arr.ndim() != sm_dim)
                    throw std::runtime_error("The SmallMatrix to create is " + std::to_string(sm_dim) +
                                             "D, but the passed array is " + std::to_string(arr.ndim()) + "D.");
                if (arr.size() != SM::column_size * SM::row_size)
                    throw std::runtime_error("Array size mismatch: Expected " + std::to_string(SM::column_size * SM::row_size) +
                                             " elements, but passed " + std::to_string(arr.size()) + " elements.");

                // TODO: check that strides are either exact or None in buf (e.g., F or C contiguous)
                // TODO: transpose if SM order is not C?

                new (self) SM{};
                auto * src = arr.data();
                std::copy(src, src + arr.size(), &self->template get<0>());

                // todo: we could check and store here if the array buffer we got is read-only

            })

            /* init from __cuda_array_interface__: device-to-host copy
             * TODO
             */


            // CPU: __array_interface__ v3
            // https://numpy.org/doc/stable/reference/arrays.interface.html
            .def_prop_ro("__array_interface__", [](SM const & sm) {
                return array_interface(sm);
            })

            // CPU: __array_function__ interface (TODO)
            //
            // NEP 18 — A dispatch mechanism for NumPy's high level array functions.
            //   https://numpy.org/neps/nep-0018-array-function-protocol.html
            // This enables code using NumPy to be directly operated on Array4 arrays.
            // __array_function__ feature requires NumPy 1.16 or later.


            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            .def_prop_ro("__cuda_array_interface__", [](SM const & sm)
            {
                auto d = array_interface(sm);

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
                d["stream"] = nb::none();

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

        ;

        // vector setter/getter
        if constexpr (column_size == 1 || row_size == 1) {
            add_get_set_Vector(py_sm);
        }

        return py_sm;
    }
}

namespace pyAMReX
{
    template<
        class T,
        int NRows,
        int NCols,
        amrex::Order ORDER = amrex::Order::F,
        int StartIndex = 0
    >
    void make_SmallMatrix (nb::module_ &m, std::string typestr)
    {
        using namespace amrex;

        using SM = SmallMatrix<T, NRows, NCols, ORDER, StartIndex>;
        using SM_T = SmallMatrix<T, NCols, NRows, ORDER, StartIndex>;  // transposed

        // Input vectors for matrix-vector and vector-matrix multiplication:
        //   SM * ColVec = (NRows x NCols) * (NCols x 1)
        //   RowVec * SM = (1 x NRows) * (NRows x NCols)
        using ColVec = SmallMatrix<T, NCols, 1, Order::F, StartIndex>;
        using RowVec = SmallMatrix<T, 1, NRows, Order::F, StartIndex>;

        nb::class_<SM> py_sm = make_SmallMatrix_or_Vector<SM>(m, typestr);
        nb::class_<ColVec> py_svc = make_SmallMatrix_or_Vector<ColVec>(m, typestr);
        nb::class_<RowVec> py_svr = make_SmallMatrix_or_Vector<RowVec>(m, typestr);

        // methods, getter, setter
        if constexpr (NRows != NCols) {
            nb::class_<SM_T> py_sm_T = make_SmallMatrix_or_Vector<SM_T>(m, typestr);
            add_matrix_methods(py_sm_T);
        }
        add_matrix_methods(py_sm);

        // operators for matrix-matrix (square matrices only)
        if constexpr (NRows == NCols) {
            py_sm
                .def(nb::self * nb::self)
            ;
        }
        // For non-square matrices, the output vectors have different
        // dimensions than the input vectors.
        //   (NRows x NCols) * (NCols x 1) = (NRows x 1)
        //   (1 x NRows) * (NRows x NCols) = (1 x NCols)
        else {
            using OutColVec = SmallMatrix<T, NRows, 1, Order::F, StartIndex>;
            using OutRowVec = SmallMatrix<T, 1, NCols, Order::F, StartIndex>;

            nb::class_<OutColVec> py_svoc = make_SmallMatrix_or_Vector<OutColVec>(m, typestr);
            nb::class_<OutRowVec> py_svor = make_SmallMatrix_or_Vector<OutRowVec>(m, typestr);

            // methods, getter, setter
            add_matrix_methods(py_svoc);
            add_matrix_methods(py_svor);
        }


        // methods, getter, setter
        add_matrix_methods(py_svc);
        add_matrix_methods(py_svr);

        // operators for matrix-vector & vector-matrix
        py_sm
            .def(nb::self * ColVec())
            .def(RowVec() * nb::self)
        ;
    }
}
