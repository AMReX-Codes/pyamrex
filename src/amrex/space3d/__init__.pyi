"""
amrex
-----
.. currentmodule:: amrex

.. autosummary::
   :toctree: _generate
   AmrCore
   AmrInfo
   AmrMesh
   AmrParGDB
   Arena
   ArrayOfStructs
   Box
   RealBox
   BoxArray
   BCRec
   BCType
   CpuBndryFuncFab
   Dim3
   FArrayBox
   iMultiFab
   IntVect
   IndexType
   RealVect
   MFInfo
   MFItInfo
   MultiFab
   ParallelDescriptor
   ParGDBBase
   Particle
   ParmParse
   ParticleTile
   ParticleContainer
   Periodicity
   PhysBCFunctNoOp
   PhysBCFunct_CpuBndryFuncFab
   PhysBCFunctUser
   PhysBCType
   PlotFileUtil
   PODVector
   SmallMatrix
   StructOfArrays
   TagBox
   TagBoxArray
   Utility
   Vector
   Vector_BCRec
   fill_domain_boundary
   setBC
   VisMF

"""

from __future__ import annotations

from amrex._dll import add_windows_dll_directories
from amrex.extensions.ParticleContainer import list_particle_species
from amrex.space3d.amrex_3d_pybind import (
    AlmostEqual,
    AmrCore,
    AMReX,
    AmrInfo,
    AmrMesh,
    AmrParGDB,
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
    ArrayOfStructs_2_1_polymorphic,
    ArrayOfStructs_16_4_arena,
    ArrayOfStructs_16_4_default,
    ArrayOfStructs_16_4_pinned,
    ArrayOfStructs_16_4_polymorphic,
    BaseFab_Real,
    BCRec,
    BCType,
    Box,
    BoxArray,
    Config,
    CoordSys,
    CpuBndryFuncFab,
    Dim3,
    Direction,
    DistributionMapping,
    DLDeviceType,
    EB2_Build,
    EBFArrayBoxFactory,
    EBSupport,
    FabArray_FArrayBox,
    FabArray_IArrayBox,
    FabArrayBase,
    FabFactory_FArrayBox,
    FabFactory_IArrayBox,
    FArrayBox,
    Geometry,
    GeometryData,
    GrowthStrategy,
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
    ParConstIter_2_1_3_1_polymorphic,
    ParConstIter_16_4_0_0_arena,
    ParConstIter_16_4_0_0_default,
    ParConstIter_16_4_0_0_pinned,
    ParConstIter_16_4_0_0_polymorphic,
    ParConstIter_pureSoA_3_0_arena,
    ParConstIter_pureSoA_3_0_default,
    ParConstIter_pureSoA_3_0_pinned,
    ParConstIter_pureSoA_3_0_polymorphic,
    ParConstIter_pureSoA_7_0_polymorphic,
    ParConstIter_pureSoA_11_0_polymorphic,
    ParConstIterBase_2_1_3_1_arena,
    ParConstIterBase_2_1_3_1_default,
    ParConstIterBase_2_1_3_1_pinned,
    ParConstIterBase_2_1_3_1_polymorphic,
    ParConstIterBase_16_4_0_0_arena,
    ParConstIterBase_16_4_0_0_default,
    ParConstIterBase_16_4_0_0_pinned,
    ParConstIterBase_16_4_0_0_polymorphic,
    ParConstIterBase_pureSoA_3_0_arena,
    ParConstIterBase_pureSoA_3_0_default,
    ParConstIterBase_pureSoA_3_0_pinned,
    ParConstIterBase_pureSoA_3_0_polymorphic,
    ParConstIterBase_pureSoA_7_0_polymorphic,
    ParConstIterBase_pureSoA_11_0_polymorphic,
    ParGDBBase,
    ParIter_2_1_3_1_arena,
    ParIter_2_1_3_1_default,
    ParIter_2_1_3_1_pinned,
    ParIter_2_1_3_1_polymorphic,
    ParIter_16_4_0_0_arena,
    ParIter_16_4_0_0_default,
    ParIter_16_4_0_0_pinned,
    ParIter_16_4_0_0_polymorphic,
    ParIter_pureSoA_3_0_arena,
    ParIter_pureSoA_3_0_default,
    ParIter_pureSoA_3_0_pinned,
    ParIter_pureSoA_3_0_polymorphic,
    ParIter_pureSoA_7_0_polymorphic,
    ParIter_pureSoA_11_0_polymorphic,
    ParIterBase_2_1_3_1_arena,
    ParIterBase_2_1_3_1_default,
    ParIterBase_2_1_3_1_pinned,
    ParIterBase_2_1_3_1_polymorphic,
    ParIterBase_16_4_0_0_arena,
    ParIterBase_16_4_0_0_default,
    ParIterBase_16_4_0_0_pinned,
    ParIterBase_16_4_0_0_polymorphic,
    ParIterBase_pureSoA_3_0_arena,
    ParIterBase_pureSoA_3_0_default,
    ParIterBase_pureSoA_3_0_pinned,
    ParIterBase_pureSoA_3_0_polymorphic,
    ParIterBase_pureSoA_7_0_polymorphic,
    ParIterBase_pureSoA_11_0_polymorphic,
    ParmParse,
    Particle_2_1,
    Particle_3_0,
    Particle_5_2,
    Particle_7_0,
    Particle_11_0,
    Particle_16_4,
    ParticleContainer_2_1_3_1_arena,
    ParticleContainer_2_1_3_1_default,
    ParticleContainer_2_1_3_1_pinned,
    ParticleContainer_2_1_3_1_polymorphic,
    ParticleContainer_16_4_0_0_arena,
    ParticleContainer_16_4_0_0_default,
    ParticleContainer_16_4_0_0_pinned,
    ParticleContainer_16_4_0_0_polymorphic,
    ParticleContainer_pureSoA_3_0_arena,
    ParticleContainer_pureSoA_3_0_default,
    ParticleContainer_pureSoA_3_0_pinned,
    ParticleContainer_pureSoA_3_0_polymorphic,
    ParticleContainer_pureSoA_7_0_polymorphic,
    ParticleContainer_pureSoA_11_0_polymorphic,
    ParticleHeader,
    ParticleInitType_2_1_3_1,
    ParticleInitType_16_4_0_0,
    ParticleInitType_pureSoA_3_0,
    ParticleInitType_pureSoA_7_0,
    ParticleInitType_pureSoA_11_0,
    ParticleTile_2_1_3_1_arena,
    ParticleTile_2_1_3_1_default,
    ParticleTile_2_1_3_1_pinned,
    ParticleTile_2_1_3_1_polymorphic,
    ParticleTile_16_4_0_0_arena,
    ParticleTile_16_4_0_0_default,
    ParticleTile_16_4_0_0_pinned,
    ParticleTile_16_4_0_0_polymorphic,
    ParticleTile_pureSoA_3_0_arena,
    ParticleTile_pureSoA_3_0_default,
    ParticleTile_pureSoA_3_0_pinned,
    ParticleTile_pureSoA_3_0_polymorphic,
    ParticleTile_pureSoA_7_0_polymorphic,
    ParticleTile_pureSoA_11_0_polymorphic,
    ParticleTileData_2_1_3_1,
    ParticleTileData_16_4_0_0,
    ParticleTileData_pureSoA_3_0,
    ParticleTileData_pureSoA_7_0,
    ParticleTileData_pureSoA_11_0,
    Periodicity,
    PhysBCFunct_CpuBndryFuncFab,
    PhysBCFunctNoOp,
    PhysBCFunctUser,
    PhysBCType,
    PlotFileData,
    PODVector_int_arena,
    PODVector_int_pinned,
    PODVector_int_polymorphic,
    PODVector_int_std,
    PODVector_real_arena,
    PODVector_real_pinned,
    PODVector_real_polymorphic,
    PODVector_real_std,
    PODVector_uint64_arena,
    PODVector_uint64_pinned,
    PODVector_uint64_polymorphic,
    PODVector_uint64_std,
    RealBox,
    RealVect,
    SmallMatrix_1x3_F_SI1_double,
    SmallMatrix_1x3_F_SI1_float,
    SmallMatrix_1x3_F_SI1_longdouble,
    SmallMatrix_1x6_F_SI1_double,
    SmallMatrix_1x6_F_SI1_float,
    SmallMatrix_1x6_F_SI1_longdouble,
    SmallMatrix_3x1_F_SI1_double,
    SmallMatrix_3x1_F_SI1_float,
    SmallMatrix_3x1_F_SI1_longdouble,
    SmallMatrix_3x6_F_SI1_double,
    SmallMatrix_3x6_F_SI1_float,
    SmallMatrix_3x6_F_SI1_longdouble,
    SmallMatrix_6x1_F_SI1_double,
    SmallMatrix_6x1_F_SI1_float,
    SmallMatrix_6x1_F_SI1_longdouble,
    SmallMatrix_6x3_F_SI1_double,
    SmallMatrix_6x3_F_SI1_float,
    SmallMatrix_6x3_F_SI1_longdouble,
    SmallMatrix_6x6_F_SI1_double,
    SmallMatrix_6x6_F_SI1_float,
    SmallMatrix_6x6_F_SI1_longdouble,
    StructOfArrays_0_0_arena,
    StructOfArrays_0_0_default,
    StructOfArrays_0_0_pinned,
    StructOfArrays_0_0_polymorphic,
    StructOfArrays_3_0_idcpu_arena,
    StructOfArrays_3_0_idcpu_default,
    StructOfArrays_3_0_idcpu_pinned,
    StructOfArrays_3_0_idcpu_polymorphic,
    StructOfArrays_3_1_arena,
    StructOfArrays_3_1_default,
    StructOfArrays_3_1_pinned,
    StructOfArrays_3_1_polymorphic,
    StructOfArrays_7_0_idcpu_polymorphic,
    StructOfArrays_11_0_idcpu_polymorphic,
    TagBox,
    TagBoxArray,
    The_Arena,
    The_Async_Arena,
    The_Cpu_Arena,
    The_Device_Arena,
    The_Managed_Arena,
    The_Pinned_Arena,
    Vector_BCRec,
    Vector_Box,
    Vector_BoxArray,
    Vector_DistributionMapping,
    Vector_Geometry,
    Vector_int,
    Vector_IntVect,
    Vector_Long,
    Vector_Real,
    Vector_string,
    VisMF,
    XDim3,
    almost_equal,
    begin,
    coarsen,
    concatenate,
    copy_mfab,
    dtoh_memcpy,
    end,
    fill_domain_boundary,
    finalize,
    htod_memcpy,
    iMultiFab,
    initialize,
    initialize_when_MPMD,
    initialized,
    is_valid,
    lbound,
    length,
    make_invalid,
    make_valid,
    makeEBFabFactory,
    max,
    min,
    pack_cpus,
    pack_ids,
    refine,
    setBC,
    size,
    ubound,
    unpack_cpus,
    unpack_ids,
    write_multi_level_plotfile,
    write_single_level_plotfile,
)
from amrex.space3d.amrex_3d_pybind import IntVect3D as IntVect
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as AsyncVector_int
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as DeviceVector_int
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as HostVector_int
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as ManagedDeviceVector_int
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as ManagedVector_int
from amrex.space3d.amrex_3d_pybind import (
    PODVector_int_std as NonManagedDeviceVector_int,
)
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as PODVector_int_default
from amrex.space3d.amrex_3d_pybind import PODVector_int_std as PinnedVector_int
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as AsyncVector_real
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as DeviceVector_real
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as HostVector_real
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as ManagedDeviceVector_real
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as ManagedVector_real
from amrex.space3d.amrex_3d_pybind import (
    PODVector_real_std as NonManagedDeviceVector_real,
)
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as PODVector_real_default
from amrex.space3d.amrex_3d_pybind import PODVector_real_std as PinnedVector_real
from amrex.space3d.amrex_3d_pybind import PODVector_uint64_std as AsyncVector_uint64
from amrex.space3d.amrex_3d_pybind import PODVector_uint64_std as DeviceVector_uint64
from amrex.space3d.amrex_3d_pybind import PODVector_uint64_std as HostVector_uint64
from amrex.space3d.amrex_3d_pybind import (
    PODVector_uint64_std as ManagedDeviceVector_uint64,
)
from amrex.space3d.amrex_3d_pybind import PODVector_uint64_std as ManagedVector_uint64
from amrex.space3d.amrex_3d_pybind import (
    PODVector_uint64_std as NonManagedDeviceVector_uint64,
)
from amrex.space3d.amrex_3d_pybind import (
    PODVector_uint64_std as PODVector_uint64_default,
)
from amrex.space3d.amrex_3d_pybind import PODVector_uint64_std as PinnedVector_uint64

from . import amrex_3d_pybind

__all__: list[str] = [
    "AMReX",
    "AlmostEqual",
    "AmrCore",
    "AmrInfo",
    "AmrMesh",
    "AmrParGDB",
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
    "ArrayOfStructs_16_4_polymorphic",
    "ArrayOfStructs_2_1_arena",
    "ArrayOfStructs_2_1_default",
    "ArrayOfStructs_2_1_pinned",
    "ArrayOfStructs_2_1_polymorphic",
    "AsyncVector_int",
    "AsyncVector_real",
    "AsyncVector_uint64",
    "BCRec",
    "BCType",
    "BaseFab_Real",
    "Box",
    "BoxArray",
    "Config",
    "CoordSys",
    "CpuBndryFuncFab",
    "DLDeviceType",
    "DeviceVector_int",
    "DeviceVector_real",
    "DeviceVector_uint64",
    "Dim3",
    "Direction",
    "DistributionMapping",
    "EB2_Build",
    "EBFArrayBoxFactory",
    "EBSupport",
    "Exact",
    "FArrayBox",
    "FabArrayBase",
    "FabArray_FArrayBox",
    "FabArray_IArrayBox",
    "FabFactory_FArrayBox",
    "FabFactory_IArrayBox",
    "Geometric",
    "Geometry",
    "GeometryData",
    "GrowthStrategy",
    "HostVector_int",
    "HostVector_real",
    "HostVector_uint64",
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
    "ManagedDeviceVector_int",
    "ManagedDeviceVector_real",
    "ManagedDeviceVector_uint64",
    "ManagedVector_int",
    "ManagedVector_real",
    "ManagedVector_uint64",
    "MultiFab",
    "NonManagedDeviceVector_int",
    "NonManagedDeviceVector_real",
    "NonManagedDeviceVector_uint64",
    "PODVector_int_arena",
    "PODVector_int_default",
    "PODVector_int_pinned",
    "PODVector_int_polymorphic",
    "PODVector_int_std",
    "PODVector_real_arena",
    "PODVector_real_default",
    "PODVector_real_pinned",
    "PODVector_real_polymorphic",
    "PODVector_real_std",
    "PODVector_uint64_arena",
    "PODVector_uint64_default",
    "PODVector_uint64_pinned",
    "PODVector_uint64_polymorphic",
    "PODVector_uint64_std",
    "ParConstIterBase_16_4_0_0_arena",
    "ParConstIterBase_16_4_0_0_default",
    "ParConstIterBase_16_4_0_0_pinned",
    "ParConstIterBase_16_4_0_0_polymorphic",
    "ParConstIterBase_2_1_3_1_arena",
    "ParConstIterBase_2_1_3_1_default",
    "ParConstIterBase_2_1_3_1_pinned",
    "ParConstIterBase_2_1_3_1_polymorphic",
    "ParConstIterBase_pureSoA_11_0_polymorphic",
    "ParConstIterBase_pureSoA_3_0_arena",
    "ParConstIterBase_pureSoA_3_0_default",
    "ParConstIterBase_pureSoA_3_0_pinned",
    "ParConstIterBase_pureSoA_3_0_polymorphic",
    "ParConstIterBase_pureSoA_7_0_polymorphic",
    "ParConstIter_16_4_0_0_arena",
    "ParConstIter_16_4_0_0_default",
    "ParConstIter_16_4_0_0_pinned",
    "ParConstIter_16_4_0_0_polymorphic",
    "ParConstIter_2_1_3_1_arena",
    "ParConstIter_2_1_3_1_default",
    "ParConstIter_2_1_3_1_pinned",
    "ParConstIter_2_1_3_1_polymorphic",
    "ParConstIter_pureSoA_11_0_polymorphic",
    "ParConstIter_pureSoA_3_0_arena",
    "ParConstIter_pureSoA_3_0_default",
    "ParConstIter_pureSoA_3_0_pinned",
    "ParConstIter_pureSoA_3_0_polymorphic",
    "ParConstIter_pureSoA_7_0_polymorphic",
    "ParGDBBase",
    "ParIterBase_16_4_0_0_arena",
    "ParIterBase_16_4_0_0_default",
    "ParIterBase_16_4_0_0_pinned",
    "ParIterBase_16_4_0_0_polymorphic",
    "ParIterBase_2_1_3_1_arena",
    "ParIterBase_2_1_3_1_default",
    "ParIterBase_2_1_3_1_pinned",
    "ParIterBase_2_1_3_1_polymorphic",
    "ParIterBase_pureSoA_11_0_polymorphic",
    "ParIterBase_pureSoA_3_0_arena",
    "ParIterBase_pureSoA_3_0_default",
    "ParIterBase_pureSoA_3_0_pinned",
    "ParIterBase_pureSoA_3_0_polymorphic",
    "ParIterBase_pureSoA_7_0_polymorphic",
    "ParIter_16_4_0_0_arena",
    "ParIter_16_4_0_0_default",
    "ParIter_16_4_0_0_pinned",
    "ParIter_16_4_0_0_polymorphic",
    "ParIter_2_1_3_1_arena",
    "ParIter_2_1_3_1_default",
    "ParIter_2_1_3_1_pinned",
    "ParIter_2_1_3_1_polymorphic",
    "ParIter_pureSoA_11_0_polymorphic",
    "ParIter_pureSoA_3_0_arena",
    "ParIter_pureSoA_3_0_default",
    "ParIter_pureSoA_3_0_pinned",
    "ParIter_pureSoA_3_0_polymorphic",
    "ParIter_pureSoA_7_0_polymorphic",
    "ParallelDescriptor",
    "ParmParse",
    "ParticleContainer_16_4_0_0_arena",
    "ParticleContainer_16_4_0_0_default",
    "ParticleContainer_16_4_0_0_pinned",
    "ParticleContainer_16_4_0_0_polymorphic",
    "ParticleContainer_2_1_3_1_arena",
    "ParticleContainer_2_1_3_1_default",
    "ParticleContainer_2_1_3_1_pinned",
    "ParticleContainer_2_1_3_1_polymorphic",
    "ParticleContainer_pureSoA_11_0_polymorphic",
    "ParticleContainer_pureSoA_3_0_arena",
    "ParticleContainer_pureSoA_3_0_default",
    "ParticleContainer_pureSoA_3_0_pinned",
    "ParticleContainer_pureSoA_3_0_polymorphic",
    "ParticleContainer_pureSoA_7_0_polymorphic",
    "ParticleHeader",
    "ParticleInitType_16_4_0_0",
    "ParticleInitType_2_1_3_1",
    "ParticleInitType_pureSoA_11_0",
    "ParticleInitType_pureSoA_3_0",
    "ParticleInitType_pureSoA_7_0",
    "ParticleTileData_16_4_0_0",
    "ParticleTileData_2_1_3_1",
    "ParticleTileData_pureSoA_11_0",
    "ParticleTileData_pureSoA_3_0",
    "ParticleTileData_pureSoA_7_0",
    "ParticleTile_16_4_0_0_arena",
    "ParticleTile_16_4_0_0_default",
    "ParticleTile_16_4_0_0_pinned",
    "ParticleTile_16_4_0_0_polymorphic",
    "ParticleTile_2_1_3_1_arena",
    "ParticleTile_2_1_3_1_default",
    "ParticleTile_2_1_3_1_pinned",
    "ParticleTile_2_1_3_1_polymorphic",
    "ParticleTile_pureSoA_11_0_polymorphic",
    "ParticleTile_pureSoA_3_0_arena",
    "ParticleTile_pureSoA_3_0_default",
    "ParticleTile_pureSoA_3_0_pinned",
    "ParticleTile_pureSoA_3_0_polymorphic",
    "ParticleTile_pureSoA_7_0_polymorphic",
    "Particle_11_0",
    "Particle_16_4",
    "Particle_2_1",
    "Particle_3_0",
    "Particle_5_2",
    "Particle_7_0",
    "Periodicity",
    "PhysBCFunctNoOp",
    "PhysBCFunctUser",
    "PhysBCFunct_CpuBndryFuncFab",
    "PhysBCType",
    "PinnedVector_int",
    "PinnedVector_real",
    "PinnedVector_uint64",
    "PlotFileData",
    "Poisson",
    "Print",
    "RealBox",
    "RealVect",
    "SmallMatrix_1x3_F_SI1_double",
    "SmallMatrix_1x3_F_SI1_float",
    "SmallMatrix_1x3_F_SI1_longdouble",
    "SmallMatrix_1x6_F_SI1_double",
    "SmallMatrix_1x6_F_SI1_float",
    "SmallMatrix_1x6_F_SI1_longdouble",
    "SmallMatrix_3x1_F_SI1_double",
    "SmallMatrix_3x1_F_SI1_float",
    "SmallMatrix_3x1_F_SI1_longdouble",
    "SmallMatrix_3x6_F_SI1_double",
    "SmallMatrix_3x6_F_SI1_float",
    "SmallMatrix_3x6_F_SI1_longdouble",
    "SmallMatrix_6x1_F_SI1_double",
    "SmallMatrix_6x1_F_SI1_float",
    "SmallMatrix_6x1_F_SI1_longdouble",
    "SmallMatrix_6x3_F_SI1_double",
    "SmallMatrix_6x3_F_SI1_float",
    "SmallMatrix_6x3_F_SI1_longdouble",
    "SmallMatrix_6x6_F_SI1_double",
    "SmallMatrix_6x6_F_SI1_float",
    "SmallMatrix_6x6_F_SI1_longdouble",
    "StructOfArrays_0_0_arena",
    "StructOfArrays_0_0_default",
    "StructOfArrays_0_0_pinned",
    "StructOfArrays_0_0_polymorphic",
    "StructOfArrays_11_0_idcpu_polymorphic",
    "StructOfArrays_3_0_idcpu_arena",
    "StructOfArrays_3_0_idcpu_default",
    "StructOfArrays_3_0_idcpu_pinned",
    "StructOfArrays_3_0_idcpu_polymorphic",
    "StructOfArrays_3_1_arena",
    "StructOfArrays_3_1_default",
    "StructOfArrays_3_1_pinned",
    "StructOfArrays_3_1_polymorphic",
    "StructOfArrays_7_0_idcpu_polymorphic",
    "TagBox",
    "TagBoxArray",
    "The_Arena",
    "The_Async_Arena",
    "The_Cpu_Arena",
    "The_Device_Arena",
    "The_Managed_Arena",
    "The_Pinned_Arena",
    "Vector_BCRec",
    "Vector_Box",
    "Vector_BoxArray",
    "Vector_DistributionMapping",
    "Vector_Geometry",
    "Vector_IntVect",
    "Vector_Long",
    "Vector_Real",
    "Vector_int",
    "Vector_string",
    "VisMF",
    "XDim3",
    "add_windows_dll_directories",
    "almost_equal",
    "amrex_3d_pybind",
    "basic",
    "begin",
    "coarsen",
    "concatenate",
    "copy_mfab",
    "d_decl",
    "dtoh_memcpy",
    "end",
    "fill_domain_boundary",
    "finalize",
    "full",
    "htod_memcpy",
    "iMultiFab",
    "initialize",
    "initialize_when_MPMD",
    "initialized",
    "is_valid",
    "kDLCPU",
    "kDLCUDA",
    "kDLCUDAHost",
    "kDLCUDAManaged",
    "kDLExtDev",
    "kDLHexagon",
    "kDLMAIA",
    "kDLMetal",
    "kDLOneAPI",
    "kDLOpenCL",
    "kDLROCM",
    "kDLROCMHost",
    "kDLVPI",
    "kDLVulkan",
    "kDLWebGPU",
    "lbound",
    "length",
    "list_particle_species",
    "makeEBFabFactory",
    "make_invalid",
    "make_valid",
    "max",
    "min",
    "pack_cpus",
    "pack_ids",
    "read_particles",
    "refine",
    "setBC",
    "size",
    "ubound",
    "unpack_cpus",
    "unpack_ids",
    "volume",
    "write_multi_level_plotfile",
    "write_single_level_plotfile",
]

def Print(*args, **kwargs):
    """
    Wrap amrex::Print() - only the IO processor writes
    """

def __getattr__(attr):
    """
    Resolve ``xp`` lazily (PEP 562).

            ``amr.xp`` is the array namespace matching this build: NumPy on CPU,
            CuPy for CUDA/HIP, dpnp for SYCL. It is the module counterpart of the
            ``to_xp`` methods, for code that needs to call into the array library
            itself, e.g. ``amr.xp.sin(...)``.

            Like every other CuPy/dpnp use in pyAMReX, those are optional
            dependencies: they are imported here on first access, never at import
            time, so ``import amrex`` works on a GPU build without them. Only
            touching ``amr.xp`` (or a ``to_cupy``/``to_dpnp``/``to_xp`` call)
            requires one to be installed.

            Raises
            ------
            ImportError
                On a GPU build whose array library (CuPy or dpnp) is not installed.

    """

def d_decl(x, y, z):
    """
    Return a tuple of the three passed elements
    """

def read_particles(
    plotfile, particle_dir="particles", communicate=True, container=None
):
    """
    Read AMReX particle data from a plotfile/checkpoint into a container.

            See :py:func:`amrex.extensions.ParticleContainer.read_particles` for details.

    """

Exact: amrex_3d_pybind.GrowthStrategy
Geometric: amrex_3d_pybind.GrowthStrategy
Poisson: amrex_3d_pybind.GrowthStrategy
__author__: str = "Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang"
__license__: str = "BSD-3-Clause-LBNL"
__version__: str = "26.08-11-g5273b558c130"
basic: amrex_3d_pybind.EBSupport
full: amrex_3d_pybind.EBSupport
kDLCPU: amrex_3d_pybind.DLDeviceType
kDLCUDA: amrex_3d_pybind.DLDeviceType
kDLCUDAHost: amrex_3d_pybind.DLDeviceType
kDLCUDAManaged: amrex_3d_pybind.DLDeviceType
kDLExtDev: amrex_3d_pybind.DLDeviceType
kDLHexagon: amrex_3d_pybind.DLDeviceType
kDLMAIA: amrex_3d_pybind.DLDeviceType
kDLMetal: amrex_3d_pybind.DLDeviceType
kDLOneAPI: amrex_3d_pybind.DLDeviceType
kDLOpenCL: amrex_3d_pybind.DLDeviceType
kDLROCM: amrex_3d_pybind.DLDeviceType
kDLROCMHost: amrex_3d_pybind.DLDeviceType
kDLVPI: amrex_3d_pybind.DLDeviceType
kDLVulkan: amrex_3d_pybind.DLDeviceType
kDLWebGPU: amrex_3d_pybind.DLDeviceType
volume: amrex_3d_pybind.EBSupport
