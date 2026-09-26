# -*- coding: utf-8 -*-

import amrex.space3d as amr


def test_interpolater_singletons():
    """The global C++ interpolater instances are exposed by reference"""
    assert isinstance(amr.pc_interp, amr.PCInterp)
    assert isinstance(amr.node_bilinear_interp, amr.NodeBilinear)
    assert isinstance(amr.face_divfree_interp, amr.FaceDivFree)
    assert isinstance(amr.face_linear_interp, amr.FaceLinear)
    assert isinstance(amr.face_cons_linear_interp, amr.FaceConservativeLinear)
    assert isinstance(amr.lincc_interp, amr.CellConservativeLinear)
    assert isinstance(amr.cell_cons_interp, amr.CellConservativeLinear)
    assert isinstance(amr.cell_bilinear_interp, amr.CellBilinear)
    assert isinstance(amr.protected_interp, amr.CellConservativeProtected)
    assert isinstance(amr.quartic_interp, amr.CellConservativeQuartic)
    assert isinstance(amr.quadratic_interp, amr.CellQuadratic)
    assert isinstance(amr.cell_quartic_interp, amr.CellQuartic)

    # every Interpolater is an InterpBase
    assert isinstance(amr.cell_cons_interp, amr.Interpolater)
    assert isinstance(amr.cell_cons_interp, amr.InterpBase)


def test_mf_interpolater_singletons():
    assert isinstance(amr.mf_pc_interp, amr.MFPCInterp)
    assert isinstance(amr.mf_cell_cons_interp, amr.MFCellConsLinInterp)
    assert isinstance(amr.mf_lincc_interp, amr.MFCellConsLinInterp)
    assert isinstance(
        amr.mf_linear_slope_minmax_interp, amr.MFCellConsLinMinmaxLimitInterp
    )
    assert isinstance(amr.mf_cell_bilinear_interp, amr.MFCellBilinear)
    assert isinstance(amr.mf_node_bilinear_interp, amr.MFNodeBilinear)

    assert isinstance(amr.mf_cell_cons_interp, amr.MFInterpolater)
    assert isinstance(amr.mf_cell_cons_interp, amr.InterpBase)


def test_singletons_identical():
    """Repeated access returns the same global instance"""
    assert amr.cell_cons_interp is amr.cell_cons_interp
