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
from amrex.extensions.Array4 import register_Array4_extension
from amrex.extensions.ArrayOfStructs import register_AoS_extension
from amrex.extensions.MultiFab import register_MultiFab_extension
from amrex.extensions.PODVector import register_PODVector_extension
from amrex.extensions.ParticleContainer import register_ParticleContainer_extension
from amrex.extensions.StructOfArrays import register_SoA_extension
from amrex.space2d.amrex_2d_pybind import AMReX
from amrex.space2d.amrex_2d_pybind import AlmostEqual
from amrex.space2d.amrex_2d_pybind import AmrInfo
from amrex.space2d.amrex_2d_pybind import AmrMesh
from amrex.space2d.amrex_2d_pybind import Arena
from amrex.space2d.amrex_2d_pybind import Array4_cdouble
from amrex.space2d.amrex_2d_pybind import Array4_cdouble_const
from amrex.space2d.amrex_2d_pybind import Array4_cfloat
from amrex.space2d.amrex_2d_pybind import Array4_cfloat_const
from amrex.space2d.amrex_2d_pybind import Array4_double
from amrex.space2d.amrex_2d_pybind import Array4_double_const
from amrex.space2d.amrex_2d_pybind import Array4_float
from amrex.space2d.amrex_2d_pybind import Array4_float_const
from amrex.space2d.amrex_2d_pybind import Array4_int
from amrex.space2d.amrex_2d_pybind import Array4_int_const
from amrex.space2d.amrex_2d_pybind import Array4_long
from amrex.space2d.amrex_2d_pybind import Array4_long_const
from amrex.space2d.amrex_2d_pybind import Array4_longdouble
from amrex.space2d.amrex_2d_pybind import Array4_longdouble_const
from amrex.space2d.amrex_2d_pybind import Array4_longlong
from amrex.space2d.amrex_2d_pybind import Array4_longlong_const
from amrex.space2d.amrex_2d_pybind import Array4_short
from amrex.space2d.amrex_2d_pybind import Array4_short_const
from amrex.space2d.amrex_2d_pybind import Array4_uint
from amrex.space2d.amrex_2d_pybind import Array4_uint_const
from amrex.space2d.amrex_2d_pybind import Array4_ulong
from amrex.space2d.amrex_2d_pybind import Array4_ulong_const
from amrex.space2d.amrex_2d_pybind import Array4_ulonglong
from amrex.space2d.amrex_2d_pybind import Array4_ulonglong_const
from amrex.space2d.amrex_2d_pybind import Array4_ushort
from amrex.space2d.amrex_2d_pybind import Array4_ushort_const
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_16_4_arena
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_16_4_default
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_16_4_pinned
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_2_1_arena
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_2_1_default
from amrex.space2d.amrex_2d_pybind import ArrayOfStructs_2_1_pinned
from amrex.space2d.amrex_2d_pybind import BaseFab_Real
from amrex.space2d.amrex_2d_pybind import Box
from amrex.space2d.amrex_2d_pybind import BoxArray
from amrex.space2d.amrex_2d_pybind import Config
from amrex.space2d.amrex_2d_pybind import CoordSys
from amrex.space2d.amrex_2d_pybind import Dim3
from amrex.space2d.amrex_2d_pybind import Direction
from amrex.space2d.amrex_2d_pybind import DistributionMapping
from amrex.space2d.amrex_2d_pybind import FArrayBox
from amrex.space2d.amrex_2d_pybind import FabArrayBase
from amrex.space2d.amrex_2d_pybind import FabArray_FArrayBox
from amrex.space2d.amrex_2d_pybind import FabFactory_FArrayBox
from amrex.space2d.amrex_2d_pybind import Geometry
from amrex.space2d.amrex_2d_pybind import GeometryData
from amrex.space2d.amrex_2d_pybind import IndexType
from amrex.space2d.amrex_2d_pybind import IntVect1D
from amrex.space2d.amrex_2d_pybind import IntVect2D as IntVect
from amrex.space2d.amrex_2d_pybind import IntVect2D
from amrex.space2d.amrex_2d_pybind import IntVect3D
from amrex.space2d.amrex_2d_pybind import MFInfo
from amrex.space2d.amrex_2d_pybind import MFItInfo
from amrex.space2d.amrex_2d_pybind import MFIter
from amrex.space2d.amrex_2d_pybind import MPMD_AppNum
from amrex.space2d.amrex_2d_pybind import MPMD_Copier
from amrex.space2d.amrex_2d_pybind import MPMD_Finalize
from amrex.space2d.amrex_2d_pybind import MPMD_Initialize_without_split
from amrex.space2d.amrex_2d_pybind import MPMD_Initialized
from amrex.space2d.amrex_2d_pybind import MPMD_MyProc
from amrex.space2d.amrex_2d_pybind import MPMD_MyProgId
from amrex.space2d.amrex_2d_pybind import MPMD_NProcs
from amrex.space2d.amrex_2d_pybind import MultiFab
from amrex.space2d.amrex_2d_pybind import PODVector_int_arena
from amrex.space2d.amrex_2d_pybind import PODVector_int_pinned
from amrex.space2d.amrex_2d_pybind import PODVector_int_std
from amrex.space2d.amrex_2d_pybind import PODVector_real_arena
from amrex.space2d.amrex_2d_pybind import PODVector_real_pinned
from amrex.space2d.amrex_2d_pybind import PODVector_real_std
from amrex.space2d.amrex_2d_pybind import PODVector_uint64_arena
from amrex.space2d.amrex_2d_pybind import PODVector_uint64_pinned
from amrex.space2d.amrex_2d_pybind import PODVector_uint64_std
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIterBase_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParConstIter_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParIterBase_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParIter_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParIter_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParIter_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParIter_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParIter_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import ParallelDescriptor
from amrex.space2d.amrex_2d_pybind import ParmParse
from amrex.space2d.amrex_2d_pybind import ParticleContainer_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleContainer_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParticleContainer_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleInitType_16_4_0_0
from amrex.space2d.amrex_2d_pybind import ParticleInitType_2_1_3_1
from amrex.space2d.amrex_2d_pybind import ParticleInitType_pureSoA_2_0
from amrex.space2d.amrex_2d_pybind import ParticleInitType_pureSoA_6_0
from amrex.space2d.amrex_2d_pybind import ParticleInitType_pureSoA_7_0
from amrex.space2d.amrex_2d_pybind import ParticleInitType_pureSoA_8_0
from amrex.space2d.amrex_2d_pybind import ParticleTileData_16_4_0_0
from amrex.space2d.amrex_2d_pybind import ParticleTileData_2_1_3_1
from amrex.space2d.amrex_2d_pybind import ParticleTileData_pureSoA_2_0
from amrex.space2d.amrex_2d_pybind import ParticleTileData_pureSoA_6_0
from amrex.space2d.amrex_2d_pybind import ParticleTileData_pureSoA_7_0
from amrex.space2d.amrex_2d_pybind import ParticleTileData_pureSoA_8_0
from amrex.space2d.amrex_2d_pybind import ParticleTile_16_4_0_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_16_4_0_0_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_16_4_0_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleTile_2_1_3_1_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_2_1_3_1_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_2_1_3_1_pinned
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_2_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_2_0_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_2_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_6_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_6_0_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_6_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_7_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_7_0_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_7_0_pinned
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_8_0_arena
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_8_0_default
from amrex.space2d.amrex_2d_pybind import ParticleTile_pureSoA_8_0_pinned
from amrex.space2d.amrex_2d_pybind import Particle_16_4
from amrex.space2d.amrex_2d_pybind import Particle_2_0
from amrex.space2d.amrex_2d_pybind import Particle_2_1
from amrex.space2d.amrex_2d_pybind import Particle_5_2
from amrex.space2d.amrex_2d_pybind import Particle_6_0
from amrex.space2d.amrex_2d_pybind import Particle_7_0
from amrex.space2d.amrex_2d_pybind import Particle_8_0
from amrex.space2d.amrex_2d_pybind import Periodicity
from amrex.space2d.amrex_2d_pybind import PlotFileData
from amrex.space2d.amrex_2d_pybind import RealBox
from amrex.space2d.amrex_2d_pybind import RealVect
from amrex.space2d.amrex_2d_pybind import StructOfArrays_0_0_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_0_0_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_0_0_pinned
from amrex.space2d.amrex_2d_pybind import StructOfArrays_2_0_idcpu_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_2_0_idcpu_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_2_0_idcpu_pinned
from amrex.space2d.amrex_2d_pybind import StructOfArrays_3_1_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_3_1_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_3_1_pinned
from amrex.space2d.amrex_2d_pybind import StructOfArrays_6_0_idcpu_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_6_0_idcpu_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_6_0_idcpu_pinned
from amrex.space2d.amrex_2d_pybind import StructOfArrays_7_0_idcpu_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_7_0_idcpu_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_7_0_idcpu_pinned
from amrex.space2d.amrex_2d_pybind import StructOfArrays_8_0_idcpu_arena
from amrex.space2d.amrex_2d_pybind import StructOfArrays_8_0_idcpu_default
from amrex.space2d.amrex_2d_pybind import StructOfArrays_8_0_idcpu_pinned
from amrex.space2d.amrex_2d_pybind import The_Arena
from amrex.space2d.amrex_2d_pybind import The_Async_Arena
from amrex.space2d.amrex_2d_pybind import The_Cpu_Arena
from amrex.space2d.amrex_2d_pybind import The_Device_Arena
from amrex.space2d.amrex_2d_pybind import The_Managed_Arena
from amrex.space2d.amrex_2d_pybind import The_Pinned_Arena
from amrex.space2d.amrex_2d_pybind import Vector_BoxArray
from amrex.space2d.amrex_2d_pybind import Vector_DistributionMapping
from amrex.space2d.amrex_2d_pybind import Vector_Geometry
from amrex.space2d.amrex_2d_pybind import Vector_IntVect
from amrex.space2d.amrex_2d_pybind import Vector_Long
from amrex.space2d.amrex_2d_pybind import Vector_Real
from amrex.space2d.amrex_2d_pybind import Vector_int
from amrex.space2d.amrex_2d_pybind import Vector_string
from amrex.space2d.amrex_2d_pybind import XDim3
from amrex.space2d.amrex_2d_pybind import begin
from amrex.space2d.amrex_2d_pybind import coarsen
from amrex.space2d.amrex_2d_pybind import concatenate
from amrex.space2d.amrex_2d_pybind import copy_mfab
from amrex.space2d.amrex_2d_pybind import dtoh_memcpy
from amrex.space2d.amrex_2d_pybind import end
from amrex.space2d.amrex_2d_pybind import finalize
from amrex.space2d.amrex_2d_pybind import htod_memcpy
from amrex.space2d.amrex_2d_pybind import initialize
from amrex.space2d.amrex_2d_pybind import initialize_when_MPMD
from amrex.space2d.amrex_2d_pybind import initialized
from amrex.space2d.amrex_2d_pybind import lbound
from amrex.space2d.amrex_2d_pybind import length
from amrex.space2d.amrex_2d_pybind import max
from amrex.space2d.amrex_2d_pybind import min
from amrex.space2d.amrex_2d_pybind import refine
from amrex.space2d.amrex_2d_pybind import size
from amrex.space2d.amrex_2d_pybind import ubound
from amrex.space2d.amrex_2d_pybind import unpack_cpus
from amrex.space2d.amrex_2d_pybind import unpack_ids
from amrex.space2d.amrex_2d_pybind import write_single_level_plotfile
import os as os
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
