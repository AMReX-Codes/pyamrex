# -*- coding: utf-8 -*-

import pytest
import pyamrex as amrex

if amrex.Config.have_mpi:
    from mpi4py import MPI

@pytest.fixture(autouse=True, scope='session')
def amrex_init():
    amrex.initialize(["amrex.verbose=-1"])
    yield
    amrex.finalize()
