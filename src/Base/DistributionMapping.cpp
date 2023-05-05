/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Vector.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <sstream>

namespace py = nanobind;
using namespace amrex;


void init_DistributionMapping(py::module_ &m) {
    py::class_< DistributionMapping >(m, "DistributionMapping")
        .def("__repr__",
            [](DistributionMapping const & dm) {
                std::stringstream s;
                s << dm.size();
                return "<amrex.DistributionMapping of size '" + s.str() + "'>";
            }
        )

        .def(py::init< >())
        .def(py::init< DistributionMapping const & >())
        //.def(py::init< DistributionMapping && >())
        //.def(py::init< DistributionMapping const &, DistributionMapping const & >())
        .def(py::init< Vector< int > const & >())
        //.def(py::init< Vector< int > && >())
        .def(py::init< BoxArray const & >(),
            py::arg("boxes")
        )
        .def(py::init< BoxArray const &, int >(),
            py::arg("boxes"), py::arg("nprocs")
        )

        .def("define",
            [](DistributionMapping & dm, BoxArray const & boxes) {
                dm.define(boxes);
            },
            py::arg("boxes")
        )
        .def("define",
            py::overload_cast< BoxArray const &, int >(&DistributionMapping::define),
            py::arg("boxes"), py::arg("nprocs")
        )
        .def("define",
            py::overload_cast< Vector< int > const & >(&DistributionMapping::define))
        //.def("define",
        //    py::overload_cast< Vector< int > && >(&DistributionMapping::define))
        //! Length of the underlying processor map.
        .def_prop_rw_readonly("size", &DistributionMapping::size)
        .def_prop_rw_readonly("capacity", &DistributionMapping::capacity)
        .def_prop_rw_readonly("empty", &DistributionMapping::empty)

        //! Number of references to this DistributionMapping
        .def_prop_rw_readonly("link_count", &DistributionMapping::linkCount)

        /**
         * \brief Returns a constant reference to the mapping of boxes in the
         * underlying BoxArray to the CPU that holds the FAB on that Box.
         * ProcessorMap()[i] is an integer in the interval [0, NCPU) where
         * NCPU is the number of CPUs being used.
         */
        .def("ProcessorMap", &DistributionMapping::ProcessorMap)

        //! Equivalent to ProcessorMap()[index].
        .def("__getitem__",
            [](DistributionMapping const & dm, int index) -> int {
                return dm[index];
            })
    ;
}
