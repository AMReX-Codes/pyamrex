# -*- coding: utf-8 -*-

import itertools
import pytest
import amrex

if amrex.Config.have_mpi:
    from mpi4py import MPI

@pytest.fixture(autouse=True, scope='session')
def amrex_init():
    amrex.initialize(["amrex.verbose=-1"])
    yield
    amrex.finalize()

@pytest.fixture(scope='module')
def boxarr():
    """BoxArray for MultiFab creation"""
    #bx = amrex.Box.new((0, 0, 0), (63, 63, 63))
    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    return ba

@pytest.fixture(scope='module')
def distmap(boxarr):
    """DistributionMapping for MultiFab creation"""
    dm = amrex.DistributionMapping(boxarr)
    return dm

@pytest.fixture(params=list(itertools.product([1,3],[0,1])))
def mfab(boxarr, distmap, request):
    """MultiFab for tests"""
    num_components = request.param[0]
    num_ghost = request.param[1]
    mfab = amrex.MultiFab(boxarr, distmap, num_components, num_ghost)
    mfab.set_val(0.0, 0, num_components)
    return mfab
