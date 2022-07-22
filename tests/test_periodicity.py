# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

def test_periodicity():
    obj = amrex.Periodicity()
    assert(obj.is_any_periodic == False)
    assert(obj.is_all_periodic == False)
    assert(obj.is_periodic(0) == False)
    assert(obj[0] == False)
    #with pytest.raises(IndexError):
    #    obj[3]

    non_periodic = amrex.Periodicity.non_periodic()
    assert(obj == non_periodic)

@pytest.mark.skipif(amrex.Config.spacedim == 3,
                    reason="Requires AMREX_SPACEDIM = 3")
def test_periodicity_3d():
    iv = amrex.IntVect(1, 0, 1)
    obj = amrex.Periodicity(iv)
    assert(obj.is_any_periodic)
    assert(obj.is_all_periodic == False)
    assert(obj.is_periodic(0))
    assert(obj.is_periodic(1) == False)
    assert(obj.is_periodic(2))
    assert(obj.is_periodic[2])

    bx = obj.domain
    print(bx)
    v_iv = ob.shift_IntVect
    print(v_iv)
