# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def test_basefab():
    bf = amr.BaseFab_Real()  # noqa


def test_basefab_to_host():
    box = amr.Box((0, 0, 0), (127, 127, 127))
    bf = amr.BaseFab_Real(box, 2, amr.The_Arena())

    host_bf = bf.to_host()
    x1 = np.array(host_bf, copy=False)
    x2 = np.array(host_bf.array(), copy=False)

    np.testing.assert_allclose(x1, x2)


def test_basefab_array4_constructor_keeps_array4_alive(
    assert_keeps_python_alive, make_real_array4
):
    arr = make_real_array4((2, 3, 4))

    assert_keeps_python_alive(arr, lambda: amr.BaseFab_Real(arr))
    assert_keeps_python_alive(
        arr, lambda: amr.BaseFab_Real(arr, amr.IndexType.cell_type())
    )


def test_basefab_dlpack(assert_keeps_python_alive):
    box = amr.Box((0, 0, 0), (7, 7, 7))
    # pinned memory is host-accessible in CPU and GPU builds alike
    bf = amr.BaseFab_Real(box, 2, amr.The_Pinned_Arena())

    view = np.from_dlpack(bf)
    assert view.shape == (2, 8, 8, 8)
    view[...] = 3.0
    assert np.all(np.from_dlpack(bf) == 3.0)

    # the consuming array keeps the BaseFab alive
    assert_keeps_python_alive(bf, lambda: np.from_dlpack(bf))


def test_basefab_dlpack_readonly():
    box = amr.Box((0, 0, 0), (3, 3, 3))
    bf = amr.BaseFab_Real(box, 1, amr.The_Pinned_Arena())

    # a const Array4 exports as a read-only tensor (DLPack >= 1.0 consumers)
    cview = np.from_dlpack(bf.const_array())
    assert not cview.flags.writeable

    # the non-const path stays writable
    view = np.from_dlpack(bf.array())
    assert view.flags.writeable
