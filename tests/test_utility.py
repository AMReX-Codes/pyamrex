# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr

def test_concatenate():
    pltname = amr.concatenate("plt",1000,5)
    print("--test concatenate --")
    print("plotfile name", pltname)
    assert (pltname == "plt01000")
