# -*- coding: utf-8 -*-
import os

import numpy as np

import amrex.space3d as amr


def test_parmparse():
    pp = amr.ParmParse("")
    dir_name = os.path.dirname(__file__)
    pp.addfile(os.path.join(dir_name, "parmparse_inputs"))
    pp_param = amr.ParmParse("param")
    _, ncell = pp_param.query_int("ncell")
    dt = pp_param.get_real("dt")
    dopml = pp_param.get_bool("do_pml")

    pp_param.add("ncell", 42)  # overwrite file
    pp_param.add(
        "question", "What is the answer to life, the universe, and everything?"
    )
    pp_param.add("answer", 41)
    pp_param.add("answer", 42)  # last wins
    pp_param.add("pi_approx", 3.1415)
    pp_param.addarr("floats", [1.0, 2.0, 3.0])
    pp_param.addarr("ints", [4, 5, 6])
    pp_param.addarr("strs", ["Who", "Where", "What", "When", "How"])

    assert dopml
    assert np.isclose(dt, 1.0e-5)
    assert ncell == 100

    # printing
    pp.pretty_print_table()

    # type hints
    d = pp.to_dict()
    assert isinstance(d["param"]["ncell"], int)  # overwritten
    assert isinstance(d["param"]["dt"], str)  # file
    assert isinstance(d["param"]["do_pml"], str)  # file
    assert isinstance(d["param"]["question"], str)
    assert isinstance(d["param"]["answer"], int)
    assert d["param"]["answer"] == 42  # last wins
    assert isinstance(d["param"]["pi_approx"], float)
    assert isinstance(d["param"]["floats"], list)
    assert isinstance(d["param"]["ints"], list)
    assert isinstance(d["param"]["strs"], list)
    assert d["param"]["floats"] == [1.0, 2.0, 3.0]
    assert d["param"]["ints"] == [4, 5, 6]
    assert d["param"]["strs"] == ["Who", "Where", "What", "When", "How"]

    # You can now dump to YAML or TOML or any other format
    # import toml
    # import yaml
    # yaml_string = yaml.dump(d)
    # toml_string = toml.dumps(d)
