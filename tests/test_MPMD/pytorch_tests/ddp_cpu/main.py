#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 The AMReX Community
#
# This file is part of AMReX.
#
# License: BSD-3-Clause-LBNL
# Authors: Revathi Jambunathan, Edoardo Zoni, Olga Shapoval, David Grote, Axel Huebl

import os
import time
import torch
import torch.distributed as dist
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

"""
Initialize the torch.distributed package.
Recommended to use backend="gloo" for CPUs
and backend="nccl" for GPUs
hostname CANNOT be localhost for multi-node runs
"""
def init_process_group(rank,world_size,backend="gloo",
        hostname="localhost"):

    # Create a TCPStore
    if(rank == 0):
        tcpstore = dist.TCPStore(host_name=hostname,
                port=12946,world_size=world_size,
                is_master = True)
    else:
        tcpstore = dist.TCPStore(host_name="localhost",
                port=12946,world_size=world_size,
                is_master = False)

    # Let us use this TCPStore in dist.init_process_group
    dist.init_process_group(backend=backend,world_size=world_size,
            rank=rank,store=tcpstore)

    if rank == 0:
        print(f"Is DDP initialized? : {dist.is_initialized()}")

"""
To broadcast a tensor from rank zero to remaining python processes
"""
def run_tensor_bcast(rank,src = 0):
    py_tensor = torch.zeros(3)
    # Populate it only if rank == src
    if rank == src:
        py_tensor[0] = 1.0
        py_tensor[1] = 2.0
        py_tensor[2] = 3.0

    print(f"py_tensor on rank-{rank} before bcast: {py_tensor}")
    # Let us now broadcast this tensor to every python process
    dist.broadcast(py_tensor,src=src)
    # Let us print the tensor on every process
    print(f"py_tensor on rank-{rank} after bcast: {py_tensor}")

# Initialize MPMD without the Split
amr.MPMD_Initialize_without_split([])

# Leverage MPI from mpi4py to perform communication split
app_comm_py = MPI.COMM_WORLD.Split(amr.MPMD_AppNum(),amr.MPMD_MyProc())

# Initialize AMReX
amr.initialize_when_MPMD([],app_comm_py)

# CPU/GPU logic
xp = load_cupy()

amr.Print(f"Hello world from pyAMReX version {amr.__version__}\n")

# Goals:
# * Define a MultiFab
# * Fill a MultiFab with data
# * Plot it

# Parameters

# Number of data components at each grid point in the MultiFab
ncomp = 2
# How many grid cells in each direction over the problem domain
n_cell = 32
# How many grid cells are allowed in each direction over each box
max_grid_size = 16

# BoxArray: abstract domain setup

# Integer vector indicating the lower coordinate bounds
dom_lo = amr.IntVect(0,0,0)
# Integer vector indicating the upper coordinate bounds
dom_hi = amr.IntVect(n_cell-1, n_cell-1, n_cell-1)
# Box containing the coordinates of this domain
domain = amr.Box(dom_lo, dom_hi)

# Will contain a list of boxes describing the problem domain
ba = amr.BoxArray(domain)

# Chop the single grid into many small boxes
ba.max_size(max_grid_size)

# Distribution Mapping
dm = amr.DistributionMapping(ba)

# Define MultiFab
mf = amr.MultiFab(ba, dm, ncomp, 0)
mf.set_val(0.)

# Geometry: physical properties for data on our domain
real_box = amr.RealBox([0., 0., 0.], [1. , 1., 1.])

coord = 0  # Cartesian
is_per = [0, 0, 0] # periodicity
geom = amr.Geometry(domain, real_box, coord, is_per)

# Calculate cell sizes
dx = geom.data().CellSize() # dx[0]=dx dx[1]=dy dx[2]=dz

# Fill a MultiFab with data
for mfi in mf:
    bx = mfi.validbox()
    # Preferred way to fill array using fast ranged operations:
    # - xp.array is indexed in reversed order (n,z,y,x),
    #   .T creates a view into the AMReX (x,y,z,n) order
    # - indices are local (range from 0 to box size)
    mf_array = xp.array(mf.array(mfi), copy=False).T
    x = (xp.arange(bx.small_end[0], bx.big_end[0]+1)+0.5)*dx[0]
    y = (xp.arange(bx.small_end[1], bx.big_end[1]+1)+0.5)*dx[1]
    z = (xp.arange(bx.small_end[2], bx.big_end[2]+1)+0.5)*dx[2]
    v = (x[xp.newaxis,xp.newaxis,:]
       + y[xp.newaxis,:,xp.newaxis]*0.1
       + z[:,xp.newaxis,xp.newaxis]*0.01)
    rsquared = ((z[xp.newaxis, xp.newaxis,          :] - 0.5)**2
              + (y[xp.newaxis,          :, xp.newaxis] - 0.5)**2
              + (x[         :, xp.newaxis, xp.newaxis] - 0.5)**2) / 0.01

    # Populate ONLY the second MultiFab component on python side
    mf_array[:, :, :, 1] = 10. + xp.exp(-rsquared)

# Create a MultiFab Copier
copr = amr.MPMD_Copier(ba,dm)
# Receive ONLY the FIRST MUltiFab component populated on cpp side
copr.recv(mf,0,1)
# Send ONLY the SECOND MUltiFab component that is populated here to cpp
copr.send(mf,1,1)

# Plot MultiFab data
plotfile = amr.concatenate(root="plt_py_", num=1, mindigits=3)
varnames = amr.Vector_string(["comp0","comp1"])
amr.write_single_level_plotfile(plotfile, mf, varnames, geom, time=0., level_step=0)

# A small pause
time.sleep(2)

# Now that data transfer is done
# Let us do distributed communication in PyTorch
init_process_group(rank=app_comm_py.Get_rank(),
        world_size=app_comm_py.Get_size())

# A small pause again
time.sleep(2)

# Let us create a tensor, but populate it only on rank zero
# We will broadcast it to remaining processes
run_tensor_bcast(rank=app_comm_py.Get_rank(),
        src=0)

# Finalize AMReX
amr.finalize()

# Finalize AMReX::MPMD
amr.MPMD_Finalize()
