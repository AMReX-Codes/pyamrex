#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 The AMReX Community
#
# This file is part of AMReX.
#
# License: BSD-3-Clause-LBNL
# Authors: Bhargav Sriram Siddani, Revathi Jambunathan, Edoardo Zoni, Olga Shapoval, David Grote, Axel Huebl

import amrex.space3d as amr
import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI

def load_cupy():
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            xp = cp
            amr.Print("Note: found and will use cupy")
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Trying managed memory in numpy...")
            import numpy as np
            xp = np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            xp = np

    else:
        import numpy as np
        xp = np
        amr.Print("Note: found and will use numpy")
    return xp

# Initialize amrex::MPMD to establish communication across the two apps
# However, leverage MPMD_Initialize_without_split
# so that communication split can be performed using mpi4py.MPI
amr.MPMD_Initialize_without_split([])
# Leverage MPI from mpi4py to perform communication split
app_comm_py = MPI.COMM_WORLD.Split(amr.MPMD_AppNum(),amr.MPMD_MyProc())
# Initialize AMReX
amr.initialize_when_MPMD([],app_comm_py)

# CPU/GPU logic
xp = load_cupy()
amr.Print(f"Hello world from pyAMReX version {amr.__version__}\n")
# Create a MPMD Copier that gets the BoxArray information from the other (C++) app
copr = amr.MPMD_Copier(True)
# Number of data components at each grid point in the MultiFab
ncomp = 2
# Define a MultiFab using the created MPMD_Copier
mf = amr.MultiFab(copr.box_array(), copr.distribution_map(), ncomp, 0)
mf.set_val(0.)

# Receive ONLY the FIRST MultiFab component populated in the other (C++) app
copr.recv(mf,0,1)

# Fill the second MultiFab component based on the first component
for mfi in mf:
    bx = mfi.validbox()
    # Preferred way to fill array using fast ranged operations:
    # - xp.array is indexed in reversed order (n,z,y,x),
    #   .T creates a view into the AMReX (x,y,z,n) order
    # - indices are local (range from 0 to box size)
    mf_array = xp.array(mf.array(mfi), copy=False).T

    mf_array[:, :, :, 1] = 10.*mf_array[:,:,:,0]

# Send ONLY the second MultiFab component to the other (C++) app
copr.send(mf,1,1)

# Plot MultiFab data
# HERE THE DOMAIN INFORMATION IS ONLY BEING UTILIZED TO SAVE A PLOTFILE
# How many grid cells in each direction over the problem domain
n_cell = 32
# Integer vector indicating the lower coordinate bounds
dom_lo = amr.IntVect(0,0,0)
# Integer vector indicating the upper coordinate bounds
dom_hi = amr.IntVect(n_cell-1, n_cell-1, n_cell-1)
# Box containing the coordinates of this domain
domain = amr.Box(dom_lo, dom_hi)
# Geometry: physical properties for data on our domain
real_box = amr.RealBox([0., 0., 0.], [1. , 1., 1.])
coord = 0  # Cartesian
is_per = [0, 0, 0] # periodicity
geom = amr.Geometry(domain, real_box, coord, is_per)
plotfile = amr.concatenate(root="plt_py_", num=1, mindigits=3)
varnames = amr.Vector_string(["comp0","comp1"])
amr.write_single_level_plotfile(plotfile, mf, varnames, geom, time=0., level_step=0)

# Finalize AMReX
amr.finalize()
# Finalize AMReX::MPMD
amr.MPMD_Finalize()
