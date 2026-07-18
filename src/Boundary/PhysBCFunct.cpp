/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Boundary/PhysBCFunct.H"

#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Vector.H>


void init_PhysBCFunct(nb::module_& m)
{
    using namespace amrex;

    nb::class_<PhysBCFunctNoOp>(m, "PhysBCFunctNoOp",
            R"pbdoc(Physical boundary condition functor that does nothing.

Use this with FillPatch-style calls when physical-domain ghost cells do
not need additional work, for example in fully periodic domains or when
the caller has already filled them.
)pbdoc")
        .def(nb::init<>(),
             "Create a no-op physical boundary functor.")
        .def("__call__", &PhysBCFunctNoOp::operator(),
             nb::arg("mf"), nb::arg("dcomp"), nb::arg("ncomp"),
             nb::arg("nghost"), nb::arg("time"), nb::arg("bccomp"),
             R"pbdoc(Apply the no-op boundary fill.

The arguments match the PhysBCFunct call interface and are accepted for
interchangeability with other physical boundary functors.

Args:
    mf: MultiFab passed by reference.
    dcomp: First destination component.
    ncomp: Number of destination components.
    nghost: Number of ghost cells to consider in each direction.
    time: Simulation time associated with the fill.
    bccomp: First boundary-condition component.
)pbdoc")
    ;

    nb::class_<CpuBndryFuncFab>(m, "CpuBndryFuncFab",
            R"pbdoc(Host boundary-fill helper for PhysBCFunct_CpuBndryFuncFab.

The default-constructed helper fills extrapolation and reflection
boundaries handled by AMReX, including BCType.foextrap,
BCType.hoextrap, BCType.hoextrapcc, BCType.reflect_even, and
BCType.reflect_odd. It leaves BCType.ext_dir and BCType.ext_dir_cc
unchanged; fill external Dirichlet values separately, for example with
PhysBCFunctUser.
)pbdoc")
        .def(nb::init<>(),
             "Create the default host boundary-fill helper.")
    ;

    nb::class_<PhysBCFunct<CpuBndryFuncFab>>(m, "PhysBCFunct_CpuBndryFuncFab",
            R"pbdoc(Physical boundary condition functor using CpuBndryFuncFab.

This wraps amrex::PhysBCFunct<CpuBndryFuncFab>. It applies the
boundary types stored in a Vector_BCRec over the physical-domain ghost
cells selected by a Geometry.
)pbdoc")
        .def(nb::init<>(),
             R"pbdoc(Create an undefined physical boundary functor.

Call define() before invoking this object.
)pbdoc")
        .def(nb::init<Geometry const &, Vector<BCRec> const &,
                      CpuBndryFuncFab const &>(),
             nb::arg("geom"), nb::arg("bc"), nb::arg("bndry_func"),
             R"pbdoc(Create a physical boundary functor.

Args:
    geom: Geometry defining the physical domain and periodic directions.
    bc: Vector_BCRec with one record per component.
    bndry_func: Boundary-fill helper, usually CpuBndryFuncFab().
)pbdoc")
        .def("define",
             nb::overload_cast<Geometry const &, Vector<BCRec> const &,
                               CpuBndryFuncFab const &>(
                 &PhysBCFunct<CpuBndryFuncFab>::define),
             nb::arg("geom"), nb::arg("bc"), nb::arg("bndry_func"),
             R"pbdoc(Reset the geometry, component BC records, and boundary helper.

Args:
    geom: Geometry defining the physical domain and periodic directions.
    bc: Vector_BCRec with one record per component.
    bndry_func: Boundary-fill helper, usually CpuBndryFuncFab().
)pbdoc")
        .def("__call__", &PhysBCFunct<CpuBndryFuncFab>::operator(),
             nb::arg("mf"), nb::arg("dcomp"), nb::arg("ncomp"),
             nb::arg("nghost"), nb::arg("time"), nb::arg("bccomp"),
             R"pbdoc(Fill physical-domain ghost cells for a component range.

Args:
    mf: MultiFab to modify in place.
    dcomp: First destination component in mf.
    ncomp: Number of components to fill.
    nghost: Number of ghost cells to consider in each direction.
    time: Simulation time associated with the fill.
    bccomp: First component in the stored Vector_BCRec that corresponds
        to dcomp.
)pbdoc")
    ;

    nb::class_<pyAMReX::PhysBCFunctUser>(m, "PhysBCFunctUser",
            R"pbdoc(Physical boundary condition functor implemented in Python.

The callback receives (mf, dcomp, ncomp, nghost, time, bccomp). It
should fill the ghost cells of mf that lie outside the physical domain
for the requested component range. This is the intended hook for
application-supplied external Dirichlet values such as BCType.ext_dir
and BCType.ext_dir_cc.

The callback runs on the host after pending AMReX GPU stream work is
synchronized. When called from C++, the wrapper acquires the Python GIL
before invoking the callback.
)pbdoc")
        .def(nb::init<>(),
             R"pbdoc(Create a user boundary functor with no callback.

Calling an empty PhysBCFunctUser is a no-op.
)pbdoc")
        .def("__init__",
             [](pyAMReX::PhysBCFunctUser * self, nb::callable callback) {
                 new (self) pyAMReX::PhysBCFunctUser(
                     [callback = nb::object(callback)](
                         MultiFab& mf, int dcomp, int ncomp,
                         IntVect const& nghost, Real time, int bccomp
                     ) {
                         nb::gil_scoped_acquire gil;
                         callback(
                             nb::cast(&mf, nb::rv_policy::reference),
                             dcomp, ncomp,
                             nb::cast(&nghost, nb::rv_policy::reference),
                             time, bccomp);
                     });
             },
             nb::arg("callback"),
             R"pbdoc(Create a user boundary functor from a Python callback.

Args:
    callback: Callable with signature
        callback(mf, dcomp, ncomp, nghost, time, bccomp).
)pbdoc")
        .def("__call__", &pyAMReX::PhysBCFunctUser::operator(),
             nb::arg("mf"), nb::arg("dcomp"), nb::arg("ncomp"),
             nb::arg("nghost"), nb::arg("time"), nb::arg("bccomp"),
             R"pbdoc(Invoke the Python physical-boundary callback.

Args:
    mf: MultiFab to modify in place.
    dcomp: First destination component in mf.
    ncomp: Number of components the callback should fill.
    nghost: Number of ghost cells to consider in each direction.
    time: Simulation time associated with the fill.
    bccomp: First boundary-condition component corresponding to dcomp.
)pbdoc")
    ;
}
