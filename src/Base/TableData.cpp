/* Copyright 2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_TableData.H>

#include <sstream>


namespace
{
    using namespace amrex;

    /** CPU: __array_interface__ v3
     *
     * https://numpy.org/doc/stable/reference/arrays.interface.html
     */
    template <class T, int N >
    py::dict
    array_interface(TableData<T, N> const & tableData)
    {
        auto d = py::dict();
        bool const read_only = false;
        d["data"] = py::make_tuple(std::intptr_t(tableData.table().p), read_only);
        d["shape"] = py::make_tuple(tableData.size());  // TODO: ND support
        d["strides"] = py::none();  // TODO: ND support
        d["typestr"] = py::format_descriptor<T>::format();
        d["version"] = 3;
        return d;
    }
}

template <class T, int N >
void make_TableData(py::module &m, std::string typestr)
{
    using namespace amrex;

    using TableData_type = TableData<T, N>;
    auto const td_name = std::string("TableData_")
                         .append(std::to_string(N)).append("D_")
                         .append(typestr);

    py::class_<TableData_type>(m, td_name.c_str())
        .def("__repr__",
             [typestr](TableData_type const & td) {
                 std::stringstream s, rs;
                 s << td.size();
                 rs << "<amrex.TableData ("
                    << N << "D) of type '"
                    << typestr << "' and size '"
                    << s.str() << "'>\n";
                 return rs.str();
             }
        )

        .def(py::init<>())
        .def(py::init<Arena*>())
        .def(py::init<Array<int, N>, Array<int, N>, Arena*>())

        // TODO: init (non-owning) from numpy arrays / buffer protocol
        // TODO: init (non-owning) from cupy arrays / cuda array protocol
        // TODO: init (non-owning) from GPU arrays / dlpack protocol

        .def_property_readonly("size", &TableData_type::size)
        .def_property_readonly("dim", &TableData_type::dim)
        .def_property_readonly("lo", &TableData_type::lo)
        .def_property_readonly("hi", &TableData_type::hi)
        .def("__len__", &TableData_type::size)

        .def("copy", &TableData_type::copy)
        .def("resize", &TableData_type::resize)
        .def("clear", &TableData_type::clear)

        //.def("table", py::overload_cast<>(&TableData_type::table))
        //.def("const_table", &TableData_type::const_table)

        .def_property_readonly("__array_interface__", [](TableData_type const & podvector) {
            return array_interface(podvector);
        })
        .def_property_readonly("__cuda_array_interface__", [](TableData_type const & podvector) {
            // Nvidia GPUs: __cuda_array_interface__ v3
            // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html
            auto d = array_interface(podvector);

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
        // TODO: setter & getter
        //.def("__setitem__", [](TableData_type & td, int const v, T const value){ td[v] = value; })
        //.def("__getitem__", [](TableData_type & td, int const v){ return td[v]; })
    ;
}

template <class T>
void make_TableData(py::module &m, std::string typestr)
{
    make_TableData<T, 1> (m, typestr);

    // TODO: ND support
    //make_TableData<T, 2> (m, typestr);
    //make_TableData<T, 3> (m, typestr);
    //make_TableData<T, 4> (m, typestr);
}

void init_TableData(py::module& m) {
    make_TableData<Real> (m, "Real");
    if constexpr(!std::is_same_v<Real, ParticleReal>)
        make_TableData<ParticleReal> (m, "ParticleReal");

    make_TableData<int> (m, "int");
    if constexpr(!std::is_same_v<int, Long>)
        make_TableData<Long> (m, "Long");

    make_TableData<uint64_t> (m, "uint64");
}
