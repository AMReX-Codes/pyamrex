# -*- coding: utf-8 -*-

import amrex.space3d as amr


def test_parmparse():
    dopml = False
    ncell = 10
    dt = 0.0
    # Since ParmParse is done in IO Processors, we need amr.initialize first
    amr.initialize([])
    pp = amr.ParmParse("")
    pp.addfile("./parmparse_inputs")
    pp_param = amr.ParmParse("param")
    (_, ncell) = pp_param.query_int("ncell")
    dt = pp_param.get_real("dt")
    dopml = pp_param.get_bool("do_pml")
    assert dopml == True
    assert dt == 1.0e-5
    assert ncell == 100
    amr.finalize()
