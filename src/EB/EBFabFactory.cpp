/* Copyright 2022 The AMReX Community
 *
 * Authors: Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFab.H>


void init_EBFabFactory (nb::module_& m)
{
    using namespace amrex;

    nb::class_<EBFArrayBoxFactory, FabFactory<FArrayBox>>(m, "EBFArrayBoxFactory")
        .def("getVolFrac", &EBFArrayBoxFactory::getVolFrac,
            nb::rv_policy::reference_internal,
            "Return volume faction MultiFab");

    nb::enum_<EBSupport>(m, "EBSupport")
        .value("basic", EBSupport::basic)
        .value("volume", EBSupport::volume)
        .value("full", EBSupport::full)
        .export_values()
    ;

    m.def(
        "makeEBFabFactory",
        [] (Geometry const& geom, BoxArray const& ba, DistributionMapping const& dm,
            Vector<int> const& ngrow, EBSupport support)
        {
            return makeEBFabFactory(geom, ba, dm, ngrow, support);
        },
        nb::arg("geom"), nb::arg("ba"), nb::arg("dm"), nb::arg("ngrow"),
        nb::arg("support"),
        "Make EBFArrayBoxFactory for given Geometry, BoxArray and DistributionMapping"
    );
}
