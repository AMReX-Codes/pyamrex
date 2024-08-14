"""

amrex
-----
.. currentmodule:: amrex

.. autosummary::
   :toctree: _generate
   AmrInfo
   AmrMesh
   Arena
   ArrayOfStructs
   Box
   RealBox
   BoxArray
   Dim3
   FArrayBox
   IntVect
   IndexType
   RealVect
   MultiFab
   ParallelDescriptor
   Particle
   ParmParse
   ParticleTile
   ParticleContainer
   Periodicity
   PlotFileUtil
   PODVector
   StructOfArrays
   Utility
   Vector

"""

from __future__ import annotations

import os as os

from amrex.extensions.Array4 import register_Array4_extension
from amrex.extensions.ArrayOfStructs import register_AoS_extension
from amrex.extensions.MultiFab import register_MultiFab_extension
from amrex.extensions.ParticleContainer import register_ParticleContainer_extension
from amrex.extensions.PODVector import register_PODVector_extension
from amrex.extensions.StructOfArrays import register_SoA_extension
from amrex.space2d.amrex_2d_pybind import (
    AlmostEqual,
    AMReX,
    AmrInfo,
    AmrMesh,
    Arena,
    Array4_cdouble,
    Array4_cdouble_const,
    Array4_cfloat,
    Array4_cfloat_const,
    Array4_double,
    Array4_double_const,
    Array4_float,
    Array4_float_const,
    Array4_int,
    Array4_int_const,
    Array4_long,
    Array4_long_const,
    Array4_longdouble,
    Array4_longdouble_const,
    Array4_longlong,
    Array4_longlong_const,
    Array4_short,
    Array4_short_const,
    Array4_uint,
    Array4_uint_const,
    Array4_ulong,
    Array4_ulong_const,
    Array4_ulonglong,
    Array4_ulonglong_const,
    Array4_ushort,
    Array4_ushort_const,
    ArrayOfStructs_2_1_arena,
    ArrayOfStructs_2_1_default,
    ArrayOfStructs_2_1_pinned,
    ArrayOfStructs_16_4_arena,
    ArrayOfStructs_16_4_default,
    ArrayOfStructs_16_4_pinned,
    BaseFab_Real,
    Box,
    BoxArray,
    Config,
    CoordSys,
    Dim3,
    Direction,
    DistributionMapping,
    FabArray_FArrayBox,
    FabArrayBase,
    FabFactory_FArrayBox,
    FArrayBox,
    Geometry,
    GeometryData,
    IndexType,
    IntVect1D,
    IntVect2D,
    IntVect3D,
    MFInfo,
    MFIter,
    MFItInfo,
    MPMD_AppNum,
    MPMD_Copier,
    MPMD_Finalize,
    MPMD_Initialize_without_split,
    MPMD_Initialized,
    MPMD_MyProc,
    MPMD_MyProgId,
    MPMD_NProcs,
    MultiFab,
    ParallelDescriptor,
    ParConstIter_2_1_3_1_arena,
    ParConstIter_2_1_3_1_default,
    ParConstIter_2_1_3_1_pinned,
    ParConstIter_16_4_0_0_arena,
    ParConstIter_16_4_0_0_default,
    ParConstIter_16_4_0_0_pinned,
    ParConstIter_pureSoA_2_0_arena,
    ParConstIter_pureSoA_2_0_default,
    ParConstIter_pureSoA_2_0_pinned,
    ParConstIter_pureSoA_6_0_arena,
    ParConstIter_pureSoA_6_0_default,
    ParConstIter_pureSoA_6_0_pinned,
    ParConstIter_pureSoA_7_0_arena,
    ParConstIter_pureSoA_7_0_default,
    ParConstIter_pureSoA_7_0_pinned,
    ParConstIter_pureSoA_8_0_arena,
    ParConstIter_pureSoA_8_0_default,
    ParConstIter_pureSoA_8_0_pinned,
    ParConstIterBase_2_1_3_1_arena,
    ParConstIterBase_2_1_3_1_default,
    ParConstIterBase_2_1_3_1_pinned,
    ParConstIterBase_16_4_0_0_arena,
    ParConstIterBase_16_4_0_0_default,
    ParConstIterBase_16_4_0_0_pinned,
    ParConstIterBase_pureSoA_2_0_arena,
    ParConstIterBase_pureSoA_2_0_default,
    ParConstIterBase_pureSoA_2_0_pinned,
    ParConstIterBase_pureSoA_6_0_arena,
    ParConstIterBase_pureSoA_6_0_default,
    ParConstIterBase_pureSoA_6_0_pinned,
    ParConstIterBase_pureSoA_7_0_arena,
    ParConstIterBase_pureSoA_7_0_default,
    ParConstIterBase_pureSoA_7_0_pinned,
    ParConstIterBase_pureSoA_8_0_arena,
    ParConstIterBase_pureSoA_8_0_default,
    ParConstIterBase_pureSoA_8_0_pinned,
    ParIter_2_1_3_1_arena,
    ParIter_2_1_3_1_default,
    ParIter_2_1_3_1_pinned,
    ParIter_16_4_0_0_arena,
    ParIter_16_4_0_0_default,
    ParIter_16_4_0_0_pinned,
    ParIter_pureSoA_2_0_arena,
    ParIter_pureSoA_2_0_default,
    ParIter_pureSoA_2_0_pinned,
    ParIter_pureSoA_6_0_arena,
    ParIter_pureSoA_6_0_default,
    ParIter_pureSoA_6_0_pinned,
    ParIter_pureSoA_7_0_arena,
    ParIter_pureSoA_7_0_default,
    ParIter_pureSoA_7_0_pinned,
    ParIter_pureSoA_8_0_arena,
    ParIter_pureSoA_8_0_default,
    ParIter_pureSoA_8_0_pinned,
    ParIterBase_2_1_3_1_arena,
    ParIterBase_2_1_3_1_default,
    ParIterBase_2_1_3_1_pinned,
    ParIterBase_16_4_0_0_arena,
    ParIterBase_16_4_0_0_default,
    ParIterBase_16_4_0_0_pinned,
    ParIterBase_pureSoA_2_0_arena,
    ParIterBase_pureSoA_2_0_default,
    ParIterBase_pureSoA_2_0_pinned,
    ParIterBase_pureSoA_6_0_arena,
    ParIterBase_pureSoA_6_0_default,
    ParIterBase_pureSoA_6_0_pinned,
    ParIterBase_pureSoA_7_0_arena,
    ParIterBase_pureSoA_7_0_default,
    ParIterBase_pureSoA_7_0_pinned,
    ParIterBase_pureSoA_8_0_arena,
    ParIterBase_pureSoA_8_0_default,
    ParIterBase_pureSoA_8_0_pinned,
    ParmParse,
    Particle_2_0,
    Particle_2_1,
    Particle_5_2,
    Particle_6_0,
    Particle_7_0,
    Particle_8_0,
    Particle_16_4,
    ParticleContainer_2_1_3_1_arena,
    ParticleContainer_2_1_3_1_default,
    ParticleContainer_2_1_3_1_pinned,
    ParticleContainer_16_4_0_0_arena,
    ParticleContainer_16_4_0_0_default,
    ParticleContainer_16_4_0_0_pinned,
    ParticleContainer_pureSoA_2_0_arena,
    ParticleContainer_pureSoA_2_0_default,
    ParticleContainer_pureSoA_2_0_pinned,
    ParticleContainer_pureSoA_6_0_arena,
    ParticleContainer_pureSoA_6_0_default,
    ParticleContainer_pureSoA_6_0_pinned,
    ParticleContainer_pureSoA_7_0_arena,
    ParticleContainer_pureSoA_7_0_default,
    ParticleContainer_pureSoA_7_0_pinned,
    ParticleContainer_pureSoA_8_0_arena,
    ParticleContainer_pureSoA_8_0_default,
    ParticleContainer_pureSoA_8_0_pinned,
    ParticleInitType_2_1_3_1,
    ParticleInitType_16_4_0_0,
    ParticleInitType_pureSoA_2_0,
    ParticleInitType_pureSoA_6_0,
    ParticleInitType_pureSoA_7_0,
    ParticleInitType_pureSoA_8_0,
    ParticleTile_2_1_3_1_arena,
    ParticleTile_2_1_3_1_default,
    ParticleTile_2_1_3_1_pinned,
    ParticleTile_16_4_0_0_arena,
    ParticleTile_16_4_0_0_default,
    ParticleTile_16_4_0_0_pinned,
    ParticleTile_pureSoA_2_0_arena,
    ParticleTile_pureSoA_2_0_default,
    ParticleTile_pureSoA_2_0_pinned,
    ParticleTile_pureSoA_6_0_arena,
    ParticleTile_pureSoA_6_0_default,
    ParticleTile_pureSoA_6_0_pinned,
    ParticleTile_pureSoA_7_0_arena,
    ParticleTile_pureSoA_7_0_default,
    ParticleTile_pureSoA_7_0_pinned,
    ParticleTile_pureSoA_8_0_arena,
    ParticleTile_pureSoA_8_0_default,
    ParticleTile_pureSoA_8_0_pinned,
    ParticleTileData_2_1_3_1,
    ParticleTileData_16_4_0_0,
    ParticleTileData_pureSoA_2_0,
    ParticleTileData_pureSoA_6_0,
    ParticleTileData_pureSoA_7_0,
    ParticleTileData_pureSoA_8_0,
    Periodicity,
    PlotFileData,
    PODVector_int_arena,
    PODVector_int_pinned,
    PODVector_int_std,
    PODVector_real_arena,
    PODVector_real_pinned,
    PODVector_real_std,
    PODVector_uint64_arena,
    PODVector_uint64_pinned,
    PODVector_uint64_std,
    RealBox,
    RealVect,
    StructOfArrays_0_0_arena,
    StructOfArrays_0_0_default,
    StructOfArrays_0_0_pinned,
    StructOfArrays_2_0_idcpu_arena,
    StructOfArrays_2_0_idcpu_default,
    StructOfArrays_2_0_idcpu_pinned,
    StructOfArrays_3_1_arena,
    StructOfArrays_3_1_default,
    StructOfArrays_3_1_pinned,
    StructOfArrays_6_0_idcpu_arena,
    StructOfArrays_6_0_idcpu_default,
    StructOfArrays_6_0_idcpu_pinned,
    StructOfArrays_7_0_idcpu_arena,
    StructOfArrays_7_0_idcpu_default,
    StructOfArrays_7_0_idcpu_pinned,
    StructOfArrays_8_0_idcpu_arena,
    StructOfArrays_8_0_idcpu_default,
    StructOfArrays_8_0_idcpu_pinned,
    The_Arena,
    The_Async_Arena,
    The_Cpu_Arena,
    The_Device_Arena,
    The_Managed_Arena,
    The_Pinned_Arena,
    Vector_BoxArray,
    Vector_DistributionMapping,
    Vector_Geometry,
    Vector_int,
    Vector_IntVect,
    Vector_Long,
    Vector_Real,
    Vector_string,
    XDim3,
    begin,
    coarsen,
    concatenate,
    copy_mfab,
    dtoh_memcpy,
    end,
    finalize,
    htod_memcpy,
    initialize,
    initialize_when_MPMD,
    initialized,
    lbound,
    length,
    max,
    min,
    refine,
    size,
    ubound,
    unpack_cpus,
    unpack_ids,
    write_single_level_plotfile,
)
from amrex.space2d.amrex_2d_pybind import IntVect2D as IntVect

from . import amrex_2d_pybind

__all__ = [
    "AMReX",
    "AlmostEqual",
    "AmrInfo",
    "AmrMesh",
    "Arena",
    "Array4_cdouble",
    "Array4_cdouble_const",
    "Array4_cfloat",
    "Array4_cfloat_const",
    "Array4_double",
    "Array4_double_const",
    "Array4_float",
    "Array4_float_const",
    "Array4_int",
    "Array4_int_const",
    "Array4_long",
    "Array4_long_const",
    "Array4_longdouble",
    "Array4_longdouble_const",
    "Array4_longlong",
    "Array4_longlong_const",
    "Array4_short",
    "Array4_short_const",
    "Array4_uint",
    "Array4_uint_const",
    "Array4_ulong",
    "Array4_ulong_const",
    "Array4_ulonglong",
    "Array4_ulonglong_const",
    "Array4_ushort",
    "Array4_ushort_const",
    "ArrayOfStructs_16_4_arena",
    "ArrayOfStructs_16_4_default",
    "ArrayOfStructs_16_4_pinned",
    "ArrayOfStructs_2_1_arena",
    "ArrayOfStructs_2_1_default",
    "ArrayOfStructs_2_1_pinned",
    "BaseFab_Real",
    "Box",
    "BoxArray",
    "Config",
    "CoordSys",
    "Dim3",
    "Direction",
    "DistributionMapping",
    "FArrayBox",
    "FabArrayBase",
    "FabArray_FArrayBox",
    "FabFactory_FArrayBox",
    "Geometry",
    "GeometryData",
    "IndexType",
    "IntVect",
    "IntVect1D",
    "IntVect2D",
    "IntVect3D",
    "MFInfo",
    "MFItInfo",
    "MFIter",
    "MPMD_AppNum",
    "MPMD_Copier",
    "MPMD_Finalize",
    "MPMD_Initialize_without_split",
    "MPMD_Initialized",
    "MPMD_MyProc",
    "MPMD_MyProgId",
    "MPMD_NProcs",
    "MultiFab",
    "PODVector_int_arena",
    "PODVector_int_pinned",
    "PODVector_int_std",
    "PODVector_real_arena",
    "PODVector_real_pinned",
    "PODVector_real_std",
    "PODVector_uint64_arena",
    "PODVector_uint64_pinned",
    "PODVector_uint64_std",
    "ParConstIterBase_16_4_0_0_arena",
    "ParConstIterBase_16_4_0_0_default",
    "ParConstIterBase_16_4_0_0_pinned",
    "ParConstIterBase_2_1_3_1_arena",
    "ParConstIterBase_2_1_3_1_default",
    "ParConstIterBase_2_1_3_1_pinned",
    "ParConstIterBase_pureSoA_2_0_arena",
    "ParConstIterBase_pureSoA_2_0_default",
    "ParConstIterBase_pureSoA_2_0_pinned",
    "ParConstIterBase_pureSoA_6_0_arena",
    "ParConstIterBase_pureSoA_6_0_default",
    "ParConstIterBase_pureSoA_6_0_pinned",
    "ParConstIterBase_pureSoA_7_0_arena",
    "ParConstIterBase_pureSoA_7_0_default",
    "ParConstIterBase_pureSoA_7_0_pinned",
    "ParConstIterBase_pureSoA_8_0_arena",
    "ParConstIterBase_pureSoA_8_0_default",
    "ParConstIterBase_pureSoA_8_0_pinned",
    "ParConstIter_16_4_0_0_arena",
    "ParConstIter_16_4_0_0_default",
    "ParConstIter_16_4_0_0_pinned",
    "ParConstIter_2_1_3_1_arena",
    "ParConstIter_2_1_3_1_default",
    "ParConstIter_2_1_3_1_pinned",
    "ParConstIter_pureSoA_2_0_arena",
    "ParConstIter_pureSoA_2_0_default",
    "ParConstIter_pureSoA_2_0_pinned",
    "ParConstIter_pureSoA_6_0_arena",
    "ParConstIter_pureSoA_6_0_default",
    "ParConstIter_pureSoA_6_0_pinned",
    "ParConstIter_pureSoA_7_0_arena",
    "ParConstIter_pureSoA_7_0_default",
    "ParConstIter_pureSoA_7_0_pinned",
    "ParConstIter_pureSoA_8_0_arena",
    "ParConstIter_pureSoA_8_0_default",
    "ParConstIter_pureSoA_8_0_pinned",
    "ParIterBase_16_4_0_0_arena",
    "ParIterBase_16_4_0_0_default",
    "ParIterBase_16_4_0_0_pinned",
    "ParIterBase_2_1_3_1_arena",
    "ParIterBase_2_1_3_1_default",
    "ParIterBase_2_1_3_1_pinned",
    "ParIterBase_pureSoA_2_0_arena",
    "ParIterBase_pureSoA_2_0_default",
    "ParIterBase_pureSoA_2_0_pinned",
    "ParIterBase_pureSoA_6_0_arena",
    "ParIterBase_pureSoA_6_0_default",
    "ParIterBase_pureSoA_6_0_pinned",
    "ParIterBase_pureSoA_7_0_arena",
    "ParIterBase_pureSoA_7_0_default",
    "ParIterBase_pureSoA_7_0_pinned",
    "ParIterBase_pureSoA_8_0_arena",
    "ParIterBase_pureSoA_8_0_default",
    "ParIterBase_pureSoA_8_0_pinned",
    "ParIter_16_4_0_0_arena",
    "ParIter_16_4_0_0_default",
    "ParIter_16_4_0_0_pinned",
    "ParIter_2_1_3_1_arena",
    "ParIter_2_1_3_1_default",
    "ParIter_2_1_3_1_pinned",
    "ParIter_pureSoA_2_0_arena",
    "ParIter_pureSoA_2_0_default",
    "ParIter_pureSoA_2_0_pinned",
    "ParIter_pureSoA_6_0_arena",
    "ParIter_pureSoA_6_0_default",
    "ParIter_pureSoA_6_0_pinned",
    "ParIter_pureSoA_7_0_arena",
    "ParIter_pureSoA_7_0_default",
    "ParIter_pureSoA_7_0_pinned",
    "ParIter_pureSoA_8_0_arena",
    "ParIter_pureSoA_8_0_default",
    "ParIter_pureSoA_8_0_pinned",
    "ParallelDescriptor",
    "ParmParse",
    "ParticleContainer_16_4_0_0_arena",
    "ParticleContainer_16_4_0_0_default",
    "ParticleContainer_16_4_0_0_pinned",
    "ParticleContainer_2_1_3_1_arena",
    "ParticleContainer_2_1_3_1_default",
    "ParticleContainer_2_1_3_1_pinned",
    "ParticleContainer_pureSoA_2_0_arena",
    "ParticleContainer_pureSoA_2_0_default",
    "ParticleContainer_pureSoA_2_0_pinned",
    "ParticleContainer_pureSoA_6_0_arena",
    "ParticleContainer_pureSoA_6_0_default",
    "ParticleContainer_pureSoA_6_0_pinned",
    "ParticleContainer_pureSoA_7_0_arena",
    "ParticleContainer_pureSoA_7_0_default",
    "ParticleContainer_pureSoA_7_0_pinned",
    "ParticleContainer_pureSoA_8_0_arena",
    "ParticleContainer_pureSoA_8_0_default",
    "ParticleContainer_pureSoA_8_0_pinned",
    "ParticleInitType_16_4_0_0",
    "ParticleInitType_2_1_3_1",
    "ParticleInitType_pureSoA_2_0",
    "ParticleInitType_pureSoA_6_0",
    "ParticleInitType_pureSoA_7_0",
    "ParticleInitType_pureSoA_8_0",
    "ParticleTileData_16_4_0_0",
    "ParticleTileData_2_1_3_1",
    "ParticleTileData_pureSoA_2_0",
    "ParticleTileData_pureSoA_6_0",
    "ParticleTileData_pureSoA_7_0",
    "ParticleTileData_pureSoA_8_0",
    "ParticleTile_16_4_0_0_arena",
    "ParticleTile_16_4_0_0_default",
    "ParticleTile_16_4_0_0_pinned",
    "ParticleTile_2_1_3_1_arena",
    "ParticleTile_2_1_3_1_default",
    "ParticleTile_2_1_3_1_pinned",
    "ParticleTile_pureSoA_2_0_arena",
    "ParticleTile_pureSoA_2_0_default",
    "ParticleTile_pureSoA_2_0_pinned",
    "ParticleTile_pureSoA_6_0_arena",
    "ParticleTile_pureSoA_6_0_default",
    "ParticleTile_pureSoA_6_0_pinned",
    "ParticleTile_pureSoA_7_0_arena",
    "ParticleTile_pureSoA_7_0_default",
    "ParticleTile_pureSoA_7_0_pinned",
    "ParticleTile_pureSoA_8_0_arena",
    "ParticleTile_pureSoA_8_0_default",
    "ParticleTile_pureSoA_8_0_pinned",
    "Particle_16_4",
    "Particle_2_0",
    "Particle_2_1",
    "Particle_5_2",
    "Particle_6_0",
    "Particle_7_0",
    "Particle_8_0",
    "Periodicity",
    "PlotFileData",
    "Print",
    "RealBox",
    "RealVect",
    "StructOfArrays_0_0_arena",
    "StructOfArrays_0_0_default",
    "StructOfArrays_0_0_pinned",
    "StructOfArrays_2_0_idcpu_arena",
    "StructOfArrays_2_0_idcpu_default",
    "StructOfArrays_2_0_idcpu_pinned",
    "StructOfArrays_3_1_arena",
    "StructOfArrays_3_1_default",
    "StructOfArrays_3_1_pinned",
    "StructOfArrays_6_0_idcpu_arena",
    "StructOfArrays_6_0_idcpu_default",
    "StructOfArrays_6_0_idcpu_pinned",
    "StructOfArrays_7_0_idcpu_arena",
    "StructOfArrays_7_0_idcpu_default",
    "StructOfArrays_7_0_idcpu_pinned",
    "StructOfArrays_8_0_idcpu_arena",
    "StructOfArrays_8_0_idcpu_default",
    "StructOfArrays_8_0_idcpu_pinned",
    "The_Arena",
    "The_Async_Arena",
    "The_Cpu_Arena",
    "The_Device_Arena",
    "The_Managed_Arena",
    "The_Pinned_Arena",
    "Vector_BoxArray",
    "Vector_DistributionMapping",
    "Vector_Geometry",
    "Vector_IntVect",
    "Vector_Long",
    "Vector_Real",
    "Vector_int",
    "Vector_string",
    "XDim3",
    "amrex_2d_pybind",
    "begin",
    "coarsen",
    "concatenate",
    "copy_mfab",
    "d_decl",
    "dtoh_memcpy",
    "end",
    "finalize",
    "htod_memcpy",
    "initialize",
    "initialize_when_MPMD",
    "initialized",
    "lbound",
    "length",
    "max",
    "min",
    "os",
    "refine",
    "register_AoS_extension",
    "register_Array4_extension",
    "register_MultiFab_extension",
    "register_PODVector_extension",
    "register_ParticleContainer_extension",
    "register_SoA_extension",
    "size",
    "ubound",
    "unpack_cpus",
    "unpack_ids",
    "write_single_level_plotfile",
]

def Print(*args, **kwargs):
    """
    Wrap amrex::Print() - only the IO processor writes
    """

def d_decl(x, y, z):
    """
    Return a tuple of the first two passed elements
    """

__author__: str = "Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang"
__license__: str = "BSD-3-Clause-LBNL"
__version__: str = "24.08-12-ge792933f9561"
