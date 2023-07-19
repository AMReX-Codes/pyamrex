# -*- coding: utf-8 -*-


import amrex.space3d as amr


def test_concatenate():
    pltname = amr.concatenate("plt", 1000, 5)
    assert pltname == "plt01000"
