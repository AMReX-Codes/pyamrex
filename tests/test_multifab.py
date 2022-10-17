# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex


def test_mfab_loop(make_mfab):
    mfab = make_mfab()
    ngv = mfab.nGrowVect
    print(f"\n  mfab={mfab}, mfab.nGrowVect={ngv}")

    for mfi in mfab:
        bx = mfi.tilebox().grow(ngv)
        marr = mfab.array(mfi)

        # print(mfab)
        # print(mfab.num_comp)
        # print(mfab.size)
        # print(marr.size)
        # print(marr.nComp)

        # index by index assignment
        # notes:
        # - this is AMReX Array4, F-order indices
        # - even though we iterate by fastest varying index,
        #   such loops are naturally very slow in Python
        three_comps = mfab.num_comp == 3
        if three_comps:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k, 0] = 10.0 * i
                marr[i, j, k, 1] = 10.0 * j
                marr[i, j, k, 2] = 10.0 * k
        else:
            for i, j, k in bx:
                # print(i,j,k)
                marr[i, j, k] = 10.0 * i

        # note: offset from index space in numpy
        #   in numpy, we start indices from zero, not small_end

        # numpy representation: non-copying view, including the
        # guard/ghost region
        #   note: in numpy, indices are in C-order!
        marr_np = np.array(marr, copy=False)

        # check the values at start/end are the same: first component
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[0, -1, -1, -1] == marr[bx.big_end]
        # same check, but for all components
        for n in range(mfab.num_comp):
            small_end_comp = list(bx.small_end) + [n]
            big_end_comp = list(bx.big_end) + [n]
            assert marr_np[n, 0, 0, 0] == marr[small_end_comp]
            assert marr_np[n, -1, -1, -1] == marr[big_end_comp]

        # now we do some faster assignments, using range based access
        #   this should fail as out-of-bounds, but does not
        #     does Numpy not check array access for non-owned views?
        # marr_np[24:200, :, :, :] = 42.

        #   all components and all indices set at once to 42
        marr_np[:, :, :, :] = 42.0

        # values in start & end still match?
        assert marr_np[0, 0, 0, 0] == marr[bx.small_end]
        assert marr_np[-1, -1, -1, -1] == marr[bx.big_end]

        # all values for all indices match between multifab & numpy view?
        for n in range(mfab.num_comp):
            for i, j, k in bx:
                assert marr[i, j, k, n] == 42.0

        # separate test: cupy assignment & reading
        #   TODO


def test_mfab_simple(make_mfab):
    mfab = make_mfab()
    assert mfab.is_all_cell_centered
    # assert(all(not mfab.is_nodal(i) for i in [-1, 0, 1, 2]))  # -1??
    assert all(not mfab.is_nodal(i) for i in [0, 1, 2])

    for i in range(mfab.num_comp):
        mfab.set_val(-10 * (i + 1), i, 1)
    mfab.abs(0, mfab.num_comp)
    for i in range(mfab.num_comp):
        assert mfab.max(i) == (10 * (i + 1))  # Assert: None == 10 for i=0
        assert mfab.min(i) == (10 * (i + 1))

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


@pytest.mark.parametrize("nghost", [0, 1])
def test_mfab_ops(boxarr, distmap, nghost):
    src = amrex.MultiFab(boxarr, distmap, 3, nghost)
    dst = amrex.MultiFab(boxarr, distmap, 1, nghost)

    src.set_val(10.0, 0, 1)
    src.set_val(20.0, 1, 1)
    src.set_val(30.0, 2, 1)
    dst.set_val(0.0, 0, 1)

    # dst.add(src, 2, 0, 1, nghost)
    # dst.subtract(src, 1, 0, 1, nghost)
    # dst.multiply(src, 0, 0, 1, nghost)
    # dst.divide(src, 1, 0, 1, nghost)

    dst.add(dst, src, 2, 0, 1, nghost)
    dst.subtract(dst, src, 1, 0, 1, nghost)
    dst.multiply(dst, src, 0, 0, 1, nghost)
    dst.divide(dst, src, 1, 0, 1, nghost)

    print(dst.min(0))
    np.testing.assert_allclose(dst.min(0), 5.0)
    np.testing.assert_allclose(dst.max(0), 5.0)

    # dst.xpay(2.0, src, 0, 0, 1, nghost)
    # dst.saxpy(2.0, src, 1, 0, 1, nghost)
    dst.xpay(dst, 2.0, src, 0, 0, 1, nghost)
    dst.saxpy(dst, 2.0, src, 1, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 60.0)
    np.testing.assert_allclose(dst.max(0), 60.0)

    # dst.lin_comb(6.0, src, 1,
    #             1.0, src, 2, 0, 1, nghost)
    dst.lin_comb(dst, 6.0, src, 1, 1.0, src, 2, 0, 1, nghost)
    np.testing.assert_allclose(dst.min(0), 150.0)
    np.testing.assert_allclose(dst.max(0), 150.0)


def test_mfab_mfiter(make_mfab):
    mfab = make_mfab()
    assert iter(mfab).is_valid
    assert iter(mfab).length == 8

    cnt = 0
    for mfi in mfab:
        cnt += 1

    assert iter(mfab).length == cnt
