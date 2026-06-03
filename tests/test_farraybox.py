# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr


def test_farraybox():
    fab = amr.FArrayBox()  # noqa


def test_farraybox_io():
    fab = amr.FArrayBox()  # noqa

    # https://docs.python.org/3/library/io.html
    # https://gist.github.com/asford/544323a5da7dddad2c9174490eb5ed06#file-test_ostream_example-py
    # import io
    # iob = io.BytesIO()
    # assert iob.getvalue() == b"..."
    # fab.read_from(iob)


def test_farraybox_array4_constructor_keeps_array4_alive(assert_keeps_python_alive):
    x = np.ones((2, 3, 4))
    arr = amr.Array4_double(x)

    assert_keeps_python_alive(arr, lambda: amr.FArrayBox(arr))
    assert_keeps_python_alive(
        arr, lambda: amr.FArrayBox(arr, amr.IndexType.cell_type())
    )
