# -*- coding: utf-8 -*-

import numpy as np
import pytest

import amrex.space3d as amr


def test_smallmatrix():
    m66 = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
        ]
    )
    v = 1
    for j in range(1, 7):
        for i in range(1, 7):
            assert m66[i, j] == v
            v += 1


def test_smallvector():
    cv1 = amr.SmallMatrix_6x1_F_SI1_double()
    rv1 = amr.SmallMatrix_1x6_F_SI1_double()
    cv2 = amr.SmallMatrix_6x1_F_SI1_double([1, 2, 3, 4, 5, 6])
    rv2 = amr.SmallMatrix_1x6_F_SI1_double([0, 10, 20, 30, 40, 50])
    cv3 = amr.SmallMatrix_6x1_F_SI1_double([0, 1, 2, 3, 4, 5])

    for j in range(1, 7):
        assert cv1[j] == 0.0
        assert rv1[j] == 0.0
        assert cv2[j] == j
        assert amr.almost_equal(rv2[j], (j - 1) * 10.0)
        assert amr.almost_equal(cv3[j], j - 1.0)


def test_smallmatrix_zero():
    zero = amr.SmallMatrix_6x6_F_SI1_double()

    # Check properties
    assert zero.size == 36
    assert zero.row_size == 6
    assert zero.column_size == 6
    assert zero.order == "F"
    assert zero.starting_index == 1

    # Check values
    assert zero.sum() == 0
    assert zero.prod() == 0
    assert zero.trace() == 0

    # assign empty
    zeroc = amr.SmallMatrix_6x6_F_SI1_double(zero)

    # Check values
    assert zeroc.sum() == 0
    assert zeroc.prod() == 0
    assert zeroc.trace() == 0

    # create zero
    zerov = amr.SmallMatrix_6x6_F_SI1_double.zero()

    # Check values
    assert zerov.sum() == 0
    assert zerov.prod() == 0
    assert zerov.trace() == 0


def test_smallmatrix_identity():
    iden = amr.SmallMatrix_6x6_F_SI1_double.identity()

    # Check properties
    assert iden.size == 36
    assert iden.row_size == 6
    assert iden.column_size == 6
    assert iden.order == "F"
    assert iden.starting_index == 1

    # Check values
    assert iden.sum() == 6
    assert iden.prod() == 0
    assert iden.trace() == 6


def test_smallmatrix_from_np():
    # from numpy (copy)
    x = np.ones(
        (
            6,
            6,
        )
    )
    print(f"\nx: {x.__array_interface__} {x.dtype}")
    sm = amr.SmallMatrix_6x6_F_SI1_double(x)
    print(f"sm: {sm.__array_interface__}")
    print(sm)

    assert sm.sum() == 36
    assert sm.prod() == 1
    assert sm.trace() == 6


def test_smallmatrix_to_np():
    iden = amr.SmallMatrix_6x6_F_SI1_double.identity()

    x = iden.to_numpy()
    print(x)

    assert x.sum() == 6
    assert x.prod() == 0
    assert x.trace() == 6
    assert not x.flags["C_CONTIGUOUS"]
    assert x.flags["F_CONTIGUOUS"]


def test_smallmatrix_smallvector():
    v3 = amr.SmallMatrix_6x1_F_SI1_double.zero()
    v3[1] = 1.0
    v3[2] = 2.0
    v3[3] = 3.0
    v3[4] = 4.0
    v3[5] = 5.0
    v3[6] = 6.0
    m66 = amr.SmallMatrix_6x6_F_SI1_double.identity()
    r = m66 * v3

    for i in range(1, 7):
        assert amr.almost_equal(r[i], v3[i])


def test_smallmatrix_smallmatrix():
    A = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [1, 0, 1, 0, 1, 0],
            [2, 1, 1, 1, 1, 2],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 2, 2, 1, 1],
            [2, 1, 2, 2, 1, 2],
            [0, 1, 1, 1, 1, 0],
        ]
    )
    B = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [1, 2, 2, 2, 1, 1],
            [2, 3, 1, 1, 1, 3],
            [4, 2, 2, 2, 2, 0],
            [1, 4, 3, 2, 0, 1],
            [2, 3, 1, 0, 0, 2],
            [0, 1, 1, 1, 4, 0],
        ]
    )
    C = amr.SmallMatrix_6x1_F_SI1_double([10, 8, 6, 4, 2, 0])
    ABC = A * B * C
    assert ABC[1, 1] == 322
    assert ABC[2, 1] == 252
    assert ABC[3, 1] == 388
    assert ABC[4, 1] == 330
    assert ABC[5, 1] == 310
    assert ABC[6, 1] == 264

    # transpose
    CR = amr.SmallMatrix_1x6_F_SI1_double([10, 8, 6, 4, 2, 0])
    ABC_T = A.T * B.transpose_in_place() * CR.T
    assert ABC_T[1, 1] == 178
    assert ABC_T[2, 1] == 402
    assert ABC_T[3, 1] == 254
    assert ABC_T[4, 1] == 476
    assert ABC_T[5, 1] == 550
    assert ABC_T[6, 1] == 254


def test_smallmatrix_sum_prod():
    m = amr.SmallMatrix_6x6_F_SI1_double()
    m.set_val(2.0)

    assert m.prod() == 2 ** (m.row_size * m.column_size)
    assert m.sum() == 2 * m.row_size * m.column_size


def test_smallmatrix_trace():
    m = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [1.0, 3.4, 4.5, 5.6, 6.7, 7.8],
            [1.3, 2.0, 3.4, 4.5, 5.6, 6.7],
            [1.3, 1.0, 3.0, 4.5, 5.6, 6.7],
            [1.3, 1.4, 4.5, 4.0, 5.6, 6.7],
            [1.3, 1.0, 4.5, 5.6, 5.0, 6.7],
            [1.3, 1.4, 3.0, 4.5, 6.7, 6.0],
        ]
    )
    assert m.trace() == 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0


def test_smallmatrix_scalar():
    A = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [+1.0, +2, +3, +4, +5, +6],
            [+7, +8, +9, +10, +11, +12],
            [+13, +14, +15, +16, +17, +18],
            [+19, +20, +21, +22, +23, +24],
            [+25, +26, +27, +28, +29, +30],
            [+31, +32, +33, +34, +35, +36],
        ]
    )
    B = amr.SmallMatrix_6x6_F_SI1_double(A)
    B *= -1.0

    # test matrix-scalar and scalar-matrix
    C = A * 2.0 + 2.0 * B
    assert np.allclose(C.to_numpy(), 0.0)

    # test unary- operator and point-wise minus
    D = -A - B
    assert np.allclose(D.to_numpy(), 0.0)

    # dot product
    E = amr.SmallMatrix_6x6_F_SI1_double()
    E.set_val(-1.0)
    assert A.dot(E) == -666


def test_smallmatrix_rangecheck():
    cv = amr.SmallMatrix_6x1_F_SI1_double()
    rv = amr.SmallMatrix_1x6_F_SI1_double()
    m66 = amr.SmallMatrix_6x6_F_SI1_double(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
        ]
    )

    with pytest.raises(RuntimeError):
        cv[0]
    with pytest.raises(RuntimeError):
        cv[7]
    with pytest.raises(RuntimeError):
        rv[0]
    with pytest.raises(RuntimeError):
        rv[7]
    with pytest.raises(RuntimeError):
        m66[0, 0]
    with pytest.raises(RuntimeError):
        m66[0, 1]
    with pytest.raises(RuntimeError):
        m66[1, 0]
    with pytest.raises(RuntimeError):
        m66[7, 7]
    with pytest.raises(RuntimeError):
        m66[6, 7]
    with pytest.raises(RuntimeError):
        m66[7, 6]
