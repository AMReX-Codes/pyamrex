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
