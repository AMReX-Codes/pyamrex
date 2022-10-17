# -*- coding: utf-8 -*-

import amrex


def test_farraybox():
    fab = amrex.FArrayBox()


def test_farraybox_io():
    fab = amrex.FArrayBox()

    # https://docs.python.org/3/library/io.html
    # https://gist.github.com/asford/544323a5da7dddad2c9174490eb5ed06#file-test_ostream_example-py
    # import io
    # iob = io.BytesIO()
    # assert iob.getvalue() == b"..."
    # fab.read_from(iob)
