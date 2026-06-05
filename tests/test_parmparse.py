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
    assert isinstance(d["param"]["dt"], float)  # file
    assert isinstance(d["param"]["do_pml"], bool)  # file
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


def test_parmparse_query_get():
    pp = amr.ParmParse("q")
    pp.add("flag", True)
    pp.add("count", 7)
    pp.add("dt", 1.5e-3)
    pp.add("name", "phi")
    pp.addarr("floats", [1.0, 2.0, 3.0])
    pp.addarr("ints", [4, 5, 6])
    pp.addarr("strs", ["a", "b"])

    # get_* of existing values
    assert pp.get_bool("flag")
    assert pp.get_int("count") == 7
    assert np.isclose(pp.get_real("dt"), 1.5e-3)
    assert pp.get_string("name") == "phi"

    # query_* of existing values
    assert pp.query_bool("flag") == (True, True)
    assert pp.query_int("count") == (True, 7)
    exists, dt = pp.query_real("dt")
    assert exists and np.isclose(dt, 1.5e-3)
    assert pp.query_string("name") == (True, "phi")

    # query_* of missing values
    assert pp.query_bool("missing")[0] is False
    assert pp.query_int("missing")[0] is False
    assert pp.query_real("missing")[0] is False
    assert pp.query_string("missing")[0] is False

    # arrays
    assert pp.countval("floats") == 3
    assert np.allclose(pp.get_real_arr("floats"), [1.0, 2.0, 3.0])
    assert pp.get_int_arr("ints") == [4, 5, 6]
    assert pp.get_string_arr("strs") == ["a", "b"]
    exists, floats = pp.query_real_arr("floats")
    assert exists and np.allclose(floats, [1.0, 2.0, 3.0])
    assert pp.query_int_arr("ints") == (True, [4, 5, 6])
    assert pp.query_string_arr("strs") == (True, ["a", "b"])
    assert pp.query_real_arr("missing")[0] is False


def test_parmparse_to_dict_prefixed():
    """to_dict is independent of the ParmParse prefix (regression test:
    this used to fail with prefixed instances)"""
    pp = amr.ParmParse("pre")
    pp.add("value", 3)

    d_root = amr.ParmParse().to_dict()
    d_pre = pp.to_dict()
    assert d_pre["pre"]["value"] == 3
    assert d_root["pre"]["value"] == 3
