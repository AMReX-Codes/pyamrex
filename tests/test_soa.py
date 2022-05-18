# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_soa_init():
    soa = amrex.StructOfArrays_2_1()
    soa.define(3,5)
    print(soa.size())

    print(soa.GetRealData())
    assert(False)