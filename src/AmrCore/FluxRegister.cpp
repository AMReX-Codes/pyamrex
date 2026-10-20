/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <sstream>


void init_FluxRegister(py::module& m)
{
    using namespace amrex;

    py::class_<FluxRegister> py_flux_register(m, "FluxRegister",
        "Stores and manipulates fluxes at coarse-fine interfaces, for "
        "conservative flux corrections (refluxing) of cell-centered data "
        "in AMR simulations.");

    py::native_enum<FluxRegister::FrOp>(py_flux_register, "FrOp",
            "enum.IntEnum",
            "Whether to copy or add fluxes into the register")
        .value("COPY", FluxRegister::COPY)
        .value("ADD", FluxRegister::ADD)
        .finalize()
    ;

    py_flux_register
        .def("__repr__",
             [](FluxRegister const & fr) {
                 std::stringstream s;
                 s << fr.nComp();
                 return "<amrex.FluxRegister with '" + s.str() +
                        "' components>";
             }
        )

        .def(py::init<>())
        .def(py::init<BoxArray const &, DistributionMapping const &,
                      IntVect const &, int, int>(),
             py::arg("fine_boxes"), py::arg("dm"), py::arg("ref_ratio"),
             py::arg("fine_lev"), py::arg("nvar"))

        .def_property_readonly("fine_level", &FluxRegister::fineLevel,
             "Returns the level number of the fine level.")
        .def_property_readonly("crse_level", &FluxRegister::crseLevel,
             "Returns the level number of the coarse level "
             "(fine_level - 1).")
        .def_property_readonly("n_comp", &FluxRegister::nComp,
             "Returns the number of components.")
        .def_property_readonly("ref_ratio", &FluxRegister::refRatio,
             "Returns the refinement ratio.")

        .def("set_val", &FluxRegister::setVal, py::arg("val"),
             "Set all registers to val.")

        .def("sum_reg", &FluxRegister::SumReg, py::arg("comp"),
             "Returns the sum over all registers of component comp.")

        .def("crse_init",
             py::overload_cast<MultiFab const &, int, int, int, int, Real,
                               FluxRegister::FrOp>(&FluxRegister::CrseInit),
             py::arg("mflx"), py::arg("dir"), py::arg("srccomp"),
             py::arg("destcomp"), py::arg("numcomp"),
             py::arg("mult") = -1.0,
             py::arg_v("op", FluxRegister::COPY, "FrOp.COPY"),
             "Initialize flux correction with coarse data (area already "
             "applied).")
        .def("crse_init",
             py::overload_cast<MultiFab const &, MultiFab const &, int, int,
                               int, int, Real, FluxRegister::FrOp>(
                 &FluxRegister::CrseInit),
             py::arg("mflx"), py::arg("area"), py::arg("dir"),
             py::arg("srccomp"), py::arg("destcomp"), py::arg("numcomp"),
             py::arg("mult") = -1.0,
             py::arg_v("op", FluxRegister::COPY, "FrOp.COPY"),
             "Initialize flux correction with coarse data and explicit "
             "face areas.")

        .def("fine_add",
             py::overload_cast<MultiFab const &, int, int, int, int, Real>(
                 &FluxRegister::FineAdd),
             py::arg("mflx"), py::arg("dir"), py::arg("srccomp"),
             py::arg("destcomp"), py::arg("numcomp"), py::arg("mult"),
             "Increment flux correction with fine data (area already "
             "applied).")
        .def("fine_add",
             py::overload_cast<MultiFab const &, MultiFab const &, int, int,
                               int, int, Real>(&FluxRegister::FineAdd),
             py::arg("mflx"), py::arg("area"), py::arg("dir"),
             py::arg("srccomp"), py::arg("destcomp"), py::arg("numcomp"),
             py::arg("mult"),
             "Increment flux correction with fine data and explicit face "
             "areas.")

        .def("reflux",
             py::overload_cast<MultiFab &, Real, int, int, int,
                               Geometry const &>(&FluxRegister::Reflux),
             py::arg("mf"), py::arg("scale"), py::arg("scomp"),
             py::arg("dcomp"), py::arg("nc"), py::arg("crse_geom"),
             "Apply the flux correction to the coarse MultiFab mf "
             "(constant-volume version).")
        .def("reflux",
             py::overload_cast<MultiFab &, MultiFab const &, Real, int, int,
                               int, Geometry const &>(&FluxRegister::Reflux),
             py::arg("mf"), py::arg("volume"), py::arg("scale"),
             py::arg("scomp"), py::arg("dcomp"), py::arg("nc"),
             py::arg("crse_geom"),
             "Apply the flux correction to the coarse MultiFab mf, with "
             "explicit cell volumes.")

        .def("clear_internal_borders",
             &FluxRegister::ClearInternalBorders, py::arg("crse_geom"),
             "Set internal borders to zero.")
    ;
}
