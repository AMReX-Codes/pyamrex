/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Interpolater.H>
#include <AMReX_InterpBase.H>
#include <AMReX_MFInterpolater.H>


void init_Interpolater(py::module& m)
{
    using namespace amrex;

    // abstract base classes: no Python constructors. Custom interpolaters
    // cannot (yet) be implemented in Python: the per-FAB interp() methods
    // are called in performance-critical inner loops.
    py::class_<InterpBase>(m, "InterpBase");
    py::class_<Interpolater, InterpBase>(m, "Interpolater",
        "Virtual base class for grid interpolaters, mapping data from "
        "coarse to fine levels.");
    py::class_<MFInterpolater, InterpBase>(m, "MFInterpolater",
        "Virtual base class for MultiFab-level grid interpolaters, "
        "mapping data from coarse to fine levels.");

    // concrete Interpolater types
    py::class_<PCInterp, Interpolater>(m, "PCInterp",
        "Piecewise constant interpolation on cell-centered data.");
    py::class_<NodeBilinear, Interpolater>(m, "NodeBilinear",
        "Bilinear interpolation on node-centered data.");
    py::class_<CellBilinear, Interpolater>(m, "CellBilinear",
        "Bilinear interpolation on cell-centered data.");
    py::class_<CellConservativeLinear, Interpolater>(
        m, "CellConservativeLinear",
        "Linear conservative interpolation on cell-centered data, i.e, "
        "conservative interpolation with a limiting scheme that "
        "preserves the value of any linear combination of the fab "
        "components.");
    py::class_<CellConservativeProtected, CellConservativeLinear>(
        m, "CellConservativeProtected",
        "Lin. cons. interp. on cc data with protection against under- or "
        "overshoots.");
    py::class_<CellConservativeQuartic, Interpolater>(
        m, "CellConservativeQuartic",
        "Quartic interpolation on cell-centered data, i.e, conservative "
        "quartic interpolation with a limiting scheme that preserves the "
        "value of any linear combination of the fab components.");
    py::class_<CellQuadratic, Interpolater>(m, "CellQuadratic",
        "Quadratic interpolation on cell-centered data.");
    py::class_<CellQuartic, Interpolater>(m, "CellQuartic",
        "Quartic interpolation on cell-centered data.");
    py::class_<FaceDivFree, Interpolater>(m, "FaceDivFree",
        "Divergence-preserving interpolation on face-centered data.");
    py::class_<FaceLinear, Interpolater>(m, "FaceLinear",
        "Piecewise constant tangential interpolation / linear normal "
        "interpolation of face data.");
    py::class_<FaceConservativeLinear, Interpolater>(
        m, "FaceConservativeLinear",
        "Bilinear tangential interpolation / linear normal interpolation "
        "of face data.");

    // concrete MFInterpolater types
    py::class_<MFPCInterp, MFInterpolater>(m, "MFPCInterp",
        "Piecewise constant interpolation on cell-centered data.");
    py::class_<MFCellConsLinInterp, MFInterpolater>(
        m, "MFCellConsLinInterp",
        "Linear conservative interpolation on cell centered data.");
    py::class_<MFCellConsLinMinmaxLimitInterp, MFInterpolater>(
        m, "MFCellConsLinMinmaxLimitInterp",
        "Linear conservative interpolation on cell centered data with "
        "the minmax limiter.");
    py::class_<MFCellBilinear, MFInterpolater>(m, "MFCellBilinear",
        "Bilinear interpolation on cell-centered data.");
    py::class_<MFNodeBilinear, MFInterpolater>(m, "MFNodeBilinear",
        "Bilinear interpolation on node-centered data.");

    // global singleton instances (defined in AMReX); handed out as
    // non-owning references
    m.attr("pc_interp") = py::cast(&pc_interp,
                                   py::return_value_policy::reference);
    m.attr("node_bilinear_interp") =
        py::cast(&node_bilinear_interp, py::return_value_policy::reference);
    m.attr("face_divfree_interp") =
        py::cast(&face_divfree_interp, py::return_value_policy::reference);
    m.attr("face_linear_interp") =
        py::cast(&face_linear_interp, py::return_value_policy::reference);
    m.attr("face_cons_linear_interp") =
        py::cast(&face_cons_linear_interp,
                 py::return_value_policy::reference);
    m.attr("lincc_interp") = py::cast(&lincc_interp,
                                      py::return_value_policy::reference);
    m.attr("cell_cons_interp") =
        py::cast(&cell_cons_interp, py::return_value_policy::reference);
    m.attr("cell_bilinear_interp") =
        py::cast(&cell_bilinear_interp, py::return_value_policy::reference);
    m.attr("protected_interp") =
        py::cast(&protected_interp, py::return_value_policy::reference);
    m.attr("quartic_interp") =
        py::cast(&quartic_interp, py::return_value_policy::reference);
    m.attr("quadratic_interp") =
        py::cast(&quadratic_interp, py::return_value_policy::reference);
    m.attr("cell_quartic_interp") =
        py::cast(&cell_quartic_interp, py::return_value_policy::reference);

    m.attr("mf_pc_interp") =
        py::cast(&mf_pc_interp, py::return_value_policy::reference);
    m.attr("mf_cell_cons_interp") =
        py::cast(&mf_cell_cons_interp, py::return_value_policy::reference);
    m.attr("mf_lincc_interp") =
        py::cast(&mf_lincc_interp, py::return_value_policy::reference);
    m.attr("mf_linear_slope_minmax_interp") =
        py::cast(&mf_linear_slope_minmax_interp,
                 py::return_value_policy::reference);
    m.attr("mf_cell_bilinear_interp") =
        py::cast(&mf_cell_bilinear_interp,
                 py::return_value_policy::reference);
    m.attr("mf_node_bilinear_interp") =
        py::cast(&mf_node_bilinear_interp,
                 py::return_value_policy::reference);
}
