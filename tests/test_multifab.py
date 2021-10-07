# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
import amrex


#@pytest.fixture(params=list(itertools.product([1,3],[0,1])))
def mfab1(boxarr, distmap, request):
    """MultiFab for tests"""
    num_components = 3 #request.param[0]
    num_ghost = 1 #request.param[1]
    mfab = amrex.MultiFab(boxarr, distmap, num_components, num_ghost)
    mfab.set_val(0.0, 0, num_components)
    return mfab # issue here when used as fixture

@pytest.mark.parametrize("nghost", [0, 1])
def test_mfab_loop(boxarr, distmap, nghost):
    mfab = amrex.MultiFab(boxarr, distmap, 3, nghost)
    # Looping
    for mfi in mfab:
        bx = mfi.tilebox()
        marr = mfab.array(mfi) # Array4: add __array_interface__ here

        for i, j, k in bx:
            mar[i, j, k, 0] = 10.0 * i
            mar[i, j, k, 1] = 10.0 * j
            mar[i, j, k, 2] = 10.0 * k
            print(i,j,k)

def test_mfab_simple(boxarr, distmap, request):
    mfab = mfab1(boxarr, distmap, request)
    assert(mfab.is_cell_centered)
    assert(all(not mfab.is_nodal(i) for i in [-1, 0, 1, 2]))

    for i in range(mfab.num_comp):
        mfab.set_val(-10 * (i + 1), i, 1)
    mfab.abs(0, mfab.num_comp)
    for i in range(mfab.num_comp):
        assert(mfab.max(i) == (10 * (i + 1))) # Assert: None == 10 for i=0
        assert(mfab.min(i) == (10 * (i + 1)))

    mfab.plus(20.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 20.0 + (10 * (i + 1)))
        np.testing.assert_allclose(mfab.min(i), 20.0 + (10 * (i + 1)))

    mfab.mult(10.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 10.0 * (20.0 + (10 * (i + 1))))
        np.testing.assert_allclose(mfab.min(i), 10.0 * (20.0 + (10 * (i + 1))))
    mfab.invert(10.0, 0, mfab.num_comp)
    for i in range(mfab.num_comp):
        np.testing.assert_allclose(mfab.max(i), 1.0 / (20.0 + (10 * (i + 1))))
        np.testing.assert_allclose(mfab.min(i), 1.0 / (20.0 + (10 * (i + 1))))
'''
@pytest.mark.parametrize("nghost", [0, 1])
def test_mfab_ops(boxarr, distmap, nghost):
    src = amrex.MultiFab(boxarr, distmap, 3, nghost)
    dst = amrex.MultiFab(boxarr, distmap, 1, nghost)

    src.set_val(10.0, 0, 1)
    src.set_val(20.0, 1, 1)
    src.set_val(30.0, 2, 1)
    dst.set_val(0.0, 0, 1)

    dst.add(src, 2, 0, 1, nghost)
    dst.subtract(src, 1, 0, 1, nghost)
    dst.multiply(src, 0, 0, 1, nghost)
    dst.divide(src, 1, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 5.0)
    np.testing.assert_allclose(dst.max(0), 5.0)

    dst.xpay(2.0, src, 0, 0, 1, nghost)
    dst.saxpy(2.0, src, 1, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 60.0)
    np.testing.assert_allclose(dst.max(0), 60.0)

    dst.lin_comb(6.0, src, 1,
                 1.0, src, 2, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 150.0)
    np.testing.assert_allclose(dst.max(0), 150.0)
'''
