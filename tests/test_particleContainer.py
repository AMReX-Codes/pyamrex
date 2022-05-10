# -*- coding: utf-8 -*-

import pytest
import numpy as np
import amrex

# @fixture
# def box():

def test_pc_init():
    pc = amrex.ParticleContainer()


    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0,0,0,1,1,1)
    coord_int = 1 # RZ
    periodicity = [0,0,0]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    pc.Define(gm,dm,ba)

    amrex.ParticleContainer(gm,dm,ba)

    assert(False)

def test_ptile():

    bx = amrex.Box(amrex.IntVect(0, 0, 0), amrex.IntVect(63, 63, 63))
    rb = amrex.RealBox(0,0,0,1,1,1)
    coord_int = 1 # RZ
    periodicity = [0,0,0]
    gm = amrex.Geometry(bx, rb, coord_int, periodicity)

    ba = amrex.BoxArray(bx)
    ba.max_size(32)
    dm = amrex.DistributionMapping(ba)

    pc = amrex.ParticleContainer(gm,dm,ba)
    pc.DefineAndReturnParticleTile()
