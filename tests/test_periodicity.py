# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def test_periodicity():
    obj = amr.Periodicity()
    assert obj.is_any_periodic == False
    assert obj.is_all_periodic == False
    assert obj.is_periodic(0) == False
    assert obj[0] == False
    # with pytest.raises(IndexError):
    #    obj[3]

    non_periodic = amr.Periodicity.non_periodic()
    assert obj == non_periodic


@pytest.mark.skipif(amr.Config.spacedim == 3, reason="Requires AMREX_SPACEDIM = 3")
def test_periodicity_3d():
    iv = amr.IntVect(1, 0, 1)
    obj = amr.Periodicity(iv)
    assert obj.is_any_periodic
    assert obj.is_all_periodic == False
    assert obj.is_periodic(0)
    assert obj.is_periodic(1) == False
    assert obj.is_periodic(2)
    assert obj.is_periodic[2]

    bx = obj.domain
    print(bx)
    v_iv = ob.shift_IntVect
    print(v_iv)
