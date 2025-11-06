
#include "pyAMReX.H"

#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFab.H>

void init_EBFabFactory (py::module& m)
{
    using namespace amrex;

    py::class_<EBFArrayBoxFactory, FabFactory<FArrayBox>>(m, "EBFArrayBoxFactory")
        .def("getVolFrac", &EBFArrayBoxFactory::getVolFrac,
            py::return_value_policy::reference_internal,
            "Return volume faction MultiFab");

    py::enum_<EBSupport>(m, "EBSupport")
        .value("basic", EBSupport::basic)
        .value("volume", EBSupport::volume)
        .value("full", EBSupport::full);

    m.def(
        "makeEBFabFactory",
        [] (Geometry const& geom, BoxArray const& ba, DistributionMapping const& dm,
            Vector<int> const& ngrow, EBSupport support)
        {
            return makeEBFabFactory(geom, ba, dm, ngrow, support);
        },
        py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("ngrow"),
        py::arg("support").
        "Make EBFArrayBoxFactory for given Geometry, BoxArray and DistributionMapping"
    );
}
