/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Boundary/PhysBCFunct.H"

#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_InterpBase.H>
#include <AMReX_MFInterpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>
#include <vector>


namespace
{
    using namespace amrex;

    /** Dispatch on the dynamic type of an InterpBase* mapper.
     *
     * The FillPatch functions are templated on the interpolater type:
     * accept any bound InterpBase in Python and dispatch to the matching
     * instantiation.
     */
    template< typename F >
    void dispatch_mapper (InterpBase* mapper, F&& f)
    {
        if (auto* interp = dynamic_cast<Interpolater*>(mapper)) {
            f(interp);
        } else if (auto* mf_interp = dynamic_cast<MFInterpolater*>(mapper)) {
            f(mf_interp);
        } else {
            throw py::value_error(
                "mapper must be an Interpolater or MFInterpolater");
        }
    }

    template< typename BC >
    void register_fill_patch (py::module& m)
    {
        m.def("fill_patch_single_level",
              [](MultiFab & mf, Real time,
                 std::vector<MultiFab*> const & smf,
                 std::vector<Real> const & stime,
                 int scomp, int dcomp, int ncomp,
                 Geometry const & geom, BC & physbcf, int bcfcomp)
              {
                  FillPatchSingleLevel(
                      mf, time,
                      Vector<MultiFab*>(smf.begin(), smf.end()),
                      Vector<Real>(stime.begin(), stime.end()),
                      scomp, dcomp, ncomp, geom, physbcf, bcfcomp);
              },
              py::arg("mf"), py::arg("time"),
              py::arg("smf"), py::arg("stime"),
              py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
              py::arg("geom"), py::arg("physbcf"), py::arg("bcfcomp"),
              "Fill mf (incl. ghost cells) with data on the same level: "
              "valid data from (time-interpolated) smf/stime source data, "
              "ghost cells from intra-level and periodic copies plus the "
              "physical boundary functor physbcf.");

        m.def("fill_patch_two_levels",
              [](MultiFab & mf, Real time,
                 std::vector<MultiFab*> const & cmf,
                 std::vector<Real> const & ct,
                 std::vector<MultiFab*> const & fmf,
                 std::vector<Real> const & ft,
                 int scomp, int dcomp, int ncomp,
                 Geometry const & cgeom, Geometry const & fgeom,
                 BC & cbc, int cbccomp, BC & fbc, int fbccomp,
                 IntVect const & ratio, InterpBase* mapper,
                 Vector<BCRec> const & bcs, int bcscomp)
              {
                  dispatch_mapper(mapper, [&](auto* interp) {
                      FillPatchTwoLevels(
                          mf, time,
                          Vector<MultiFab*>(cmf.begin(), cmf.end()),
                          Vector<Real>(ct.begin(), ct.end()),
                          Vector<MultiFab*>(fmf.begin(), fmf.end()),
                          Vector<Real>(ft.begin(), ft.end()),
                          scomp, dcomp, ncomp, cgeom, fgeom,
                          cbc, cbccomp, fbc, fbccomp, ratio, interp,
                          bcs, bcscomp);
                  });
              },
              py::arg("mf"), py::arg("time"),
              py::arg("cmf"), py::arg("ct"), py::arg("fmf"), py::arg("ft"),
              py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
              py::arg("cgeom"), py::arg("fgeom"),
              py::arg("cbc"), py::arg("cbccomp"),
              py::arg("fbc"), py::arg("fbccomp"),
              py::arg("ratio"), py::arg("mapper"),
              py::arg("bcs"), py::arg("bcscomp"),
              "Fill mf (incl. ghost cells) with data from the fine level "
              "it lives on (fmf/ft) where possible, and spatially "
              "interpolated (mapper, with bcs boundary conditions) from "
              "the coarse level (cmf/ct) elsewhere. Includes time "
              "interpolation of the source data.");

        m.def("interp_from_coarse_level",
              [](MultiFab & mf, Real time, MultiFab const & cmf,
                 int scomp, int dcomp, int ncomp,
                 Geometry const & cgeom, Geometry const & fgeom,
                 BC & cbc, int cbccomp, BC & fbc, int fbccomp,
                 IntVect const & ratio, InterpBase* mapper,
                 Vector<BCRec> const & bcs, int bcscomp)
              {
                  dispatch_mapper(mapper, [&](auto* interp) {
                      InterpFromCoarseLevel(
                          mf, time, cmf, scomp, dcomp, ncomp, cgeom, fgeom,
                          cbc, cbccomp, fbc, fbccomp, ratio, interp,
                          bcs, bcscomp);
                  });
              },
              py::arg("mf"), py::arg("time"), py::arg("cmf"),
              py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
              py::arg("cgeom"), py::arg("fgeom"),
              py::arg("cbc"), py::arg("cbccomp"),
              py::arg("fbc"), py::arg("fbccomp"),
              py::arg("ratio"), py::arg("mapper"),
              py::arg("bcs"), py::arg("bcscomp"),
              "Fill mf (incl. ghost cells) entirely by spatial "
              "interpolation (mapper, with bcs boundary conditions) from "
              "the coarse level data cmf. This comes into play, e.g., "
              "when a new level of refinement appears.");
    }
}


void init_FillPatchUtil(py::module& m)
{
    using namespace amrex;

    // overloads for each bound physical boundary condition functor type
    register_fill_patch< PhysBCFunctNoOp >(m);
    register_fill_patch< PhysBCFunct<CpuBndryFuncFab> >(m);
    register_fill_patch< pyAMReX::PhysBCFunctUser >(m);
}
