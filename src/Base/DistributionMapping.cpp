/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Vector.H>

#include <sstream>


void init_DistributionMapping(nb::module_ &m) {
    using namespace amrex;

    nb::class_< DistributionMapping >(m, "DistributionMapping")
        .def("__repr__",
            [](DistributionMapping const & dm) {
                std::stringstream s;
                s << dm.size();
                return "<amrex.DistributionMapping of size '" + s.str() + "'>";
            }
        )

        .def(nb::init< >())
        .def(nb::init< DistributionMapping const & >())
        //.def(nb::init< DistributionMapping && >())
        //.def(nb::init< DistributionMapping const &, DistributionMapping const & >())
        .def(nb::init< Vector< int > const & >())
        //.def(nb::init< Vector< int > && >())
        .def(nb::init< BoxArray const & >(),
            nb::arg("boxes")
        )
        .def(nb::init< BoxArray const &, int >(),
            nb::arg("boxes"), nb::arg("nprocs")
        )

        .def("define",
            [](DistributionMapping & dm, BoxArray const & boxes) {
                dm.define(boxes);
            },
            nb::arg("boxes")
        )
        .def("define",
            nb::overload_cast< BoxArray const &, int >(&DistributionMapping::define),
            nb::arg("boxes"), nb::arg("nprocs")
        )
        .def("define",
            nb::overload_cast< Vector< int > const & >(&DistributionMapping::define))
        //.def("define",
        //    nb::overload_cast< Vector< int > && >(&DistributionMapping::define))
        //! Length of the underlying processor map.
        .def_prop_ro("size", &DistributionMapping::size)
        .def_prop_ro("capacity", &DistributionMapping::capacity)
        .def_prop_ro("empty", &DistributionMapping::empty)

        //! Number of references to this DistributionMapping
        .def_prop_ro("link_count", &DistributionMapping::linkCount)

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

    make_Vector<DistributionMapping> (m, "DistributionMapping");
}
