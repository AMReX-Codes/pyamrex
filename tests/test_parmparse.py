# -*- coding: utf-8 -*-
import os

import amrex.space3d as amr


def test_parmparse():
    pp = amr.ParmParse("")
    dir_name = os.path.dirname(__file__)
    pp.addfile(os.path.join(dir_name, "/parmparse_inputs"))
    pp_param = amr.ParmParse("param")
    _, ncell = pp_param.query_int("ncell")
    dt = pp_param.get_real("dt")
    dopml = pp_param.get_bool("do_pml")

    assert dopml == True
    assert dt == 1.0e-5
    assert ncell == 100
