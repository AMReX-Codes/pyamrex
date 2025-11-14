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
   iMultiFab
   IntVect
   IndexType
   RealVect
   MFInfo
   MFItInfo
   MultiFab
   ParallelDescriptor
   Particle
   ParmParse
   ParticleTile
   ParticleContainer
   Periodicity
   PlotFileUtil
   PODVector
   SmallMatrix
   StructOfArrays
   Utility
   Vector
   VisMF

"""

from __future__ import annotations

import collections.abc
import enum
import typing

import numpy
import numpy.typing

import amrex.space2d

from . import ParallelDescriptor

__all__: list[str] = [
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
    "Poisson",
    "RealBox",
    "RealVect",
    "SmallMatrix_1x6_F_SI1_double",
    "SmallMatrix_1x6_F_SI1_float",
    "SmallMatrix_1x6_F_SI1_longdouble",
    "SmallMatrix_6x1_F_SI1_double",
    "SmallMatrix_6x1_F_SI1_float",
    "SmallMatrix_6x1_F_SI1_longdouble",
    "SmallMatrix_6x6_F_SI1_double",
    "SmallMatrix_6x6_F_SI1_float",
    "SmallMatrix_6x6_F_SI1_longdouble",
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
    "almost_equal",
    "basic",
    "begin",
    "coarsen",
    "concatenate",
    "copy_mfab",
    "dtoh_memcpy",
    "end",
    "finalize",
    "full",
    "htod_memcpy",
    "iMultiFab",
    "initialize",
    "initialize_when_MPMD",
    "initialized",
    "is_valid",
    "lbound",
    "length",
    "makeEBFabFactory",
    "make_invalid",
    "make_valid",
    "max",
    "min",
    "pack_cpus",
    "pack_ids",
    "refine",
    "size",
    "ubound",
    "unpack_cpus",
    "unpack_ids",
    "volume",
    "write_single_level_plotfile",
]

class AMReX:
    @staticmethod
    def empty() -> bool: ...
    @staticmethod
    def erase(arg0: AMReX) -> None: ...
    @staticmethod
    def size() -> int: ...
    @staticmethod
    def top() -> AMReX: ...

class AmrInfo:
    check_input: bool
    iterate_on_new_grids: bool
    refine_grid_layout: bool
    refine_grid_layout_dims: IntVect2D
    use_fixed_coarse_grids: bool
    use_new_chop: bool
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def blocking_factor(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    def max_grid_size(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    def n_error_buf(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    def ref_ratio(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @property
    def grid_eff(self) -> float: ...
    @grid_eff.setter
    def grid_eff(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def max_level(self) -> int: ...
    @max_level.setter
    def max_level(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def n_proper(self) -> int: ...
    @n_proper.setter
    def n_proper(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def use_fixed_upto_level(self) -> int: ...
    @use_fixed_upto_level.setter
    def use_fixed_upto_level(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def verbose(self) -> int: ...
    @verbose.setter
    def verbose(self, arg0: typing.SupportsInt) -> None: ...

class AmrMesh:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        rb: RealBox,
        max_level_in: typing.SupportsInt,
        n_cell_in: Vector_int,
        coord: typing.SupportsInt,
        ref_ratios: Vector_IntVect,
        is_per: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def ref_ratio(self) -> Vector_IntVect: ...
    @typing.overload
    def ref_ratio(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def max_level(self) -> int: ...
    @property
    def verbose(self) -> int: ...

class Arena:
    @staticmethod
    def finalize() -> None: ...
    @staticmethod
    def initialize(arg0: bool) -> None: ...
    @staticmethod
    def print_usage(arg0: bool) -> None: ...
    @staticmethod
    def print_usage_to_files(filename: str, message: str) -> None: ...
    def has_free_device_memory(self, sz: typing.SupportsInt) -> bool:
        """
        Does the device have enough free memory for allocating this much memory? For CPU builds, this always return true.
        """
    @property
    def is_device(self) -> bool: ...
    @property
    def is_device_accessible(self) -> bool: ...
    @property
    def is_host_accessible(self) -> bool: ...
    @property
    def is_managed(self) -> bool: ...
    @property
    def is_pinned(self) -> bool: ...

class Array4_cdouble:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_cdouble, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: complex,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: complex,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.complex128]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_cdouble_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble_const) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_cdouble_const, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_cdouble_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, complex]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.complex128]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_cfloat:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_cfloat, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: complex,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: complex,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.complex64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_cfloat_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> complex: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_cfloat_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, complex]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.complex64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_double:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_double, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_double_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_double_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_float:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_float, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.float32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_float_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_float_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.float32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_int:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_int, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_int_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_int_const, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_long:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_long, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_long_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_long_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_longdouble:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_longdouble,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.longdouble]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.longdouble]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_longdouble_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble_const) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_longdouble_const, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_longdouble_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.longdouble]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.longdouble]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_longlong:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_longlong, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_longlong_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong_const) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_longlong_const, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_longlong_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_short:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_short, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int16]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_short_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_short_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int16]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.int16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_uint:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_uint, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_uint_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_uint_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ulong:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_ulong, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ulong_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_ulong_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ulonglong:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_ulonglong, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ulonglong_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong_const) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_ulonglong_const, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_ulonglong_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ushort:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Array4_ushort, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint16]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
        arg1: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class Array4_ushort_const:
    @typing.overload
    def __getitem__(self, arg0: IntVect2D) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> int: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort_const, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Array4_ushort_const,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint16]
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def contains(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> bool: ...
    @typing.overload
    def contains(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def contains(self, arg0: Dim3) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy n-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> numpy.typing.NDArray[numpy.uint16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an Array4.

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy n-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into an Array4, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of the box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.Array4_*
            An Array4 class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy n-dimensional array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def nComp(self) -> int: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class ArrayOfStructs_16_4_arena:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def back(self) -> Particle_16_4:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_16_4) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_16_4_default:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def back(self) -> Particle_16_4:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_16_4) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_16_4_pinned:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def back(self) -> Particle_16_4:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_16_4) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_arena:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_1) -> None: ...
    def back(self) -> Particle_2_1:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_2_1) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_default:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_1) -> None: ...
    def back(self) -> Particle_2_1:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_2_1) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_pinned:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_1) -> None: ...
    def back(self) -> Particle_2_1:
        """
        get back member.  Problem!!!!! this is perfo
        """
    @typing.overload
    def empty(self) -> bool: ...
    @typing.overload
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: Particle_2_1) -> None: ...
    def setNumNeighbors(self, arg0: typing.SupportsInt) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a ArrayOfStructs.

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy arrays.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a ArrayOfStructs, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.ArrayOfStructs_*
            An ArrayOfStructs class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D NumPy or CuPy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class BaseFab_Real:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Arena) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: typing.SupportsInt, arg2: Arena) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: typing.SupportsInt, arg2: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: typing.SupportsInt, arg2: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double, arg1: IndexType) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const, arg1: IndexType) -> None: ...
    def __repr__(self) -> str: ...
    def array(self) -> Array4_double: ...
    def big_end(self) -> IntVect2D: ...
    def box(self) -> Box: ...
    def clear(self) -> None: ...
    def const_array(self) -> Array4_double_const: ...
    def hi_vect(self) -> int: ...
    def is_allocated(self) -> bool: ...
    def length(self) -> IntVect2D: ...
    def lo_vect(self) -> int: ...
    @typing.overload
    def n_bytes(self) -> int: ...
    @typing.overload
    def n_bytes(self, arg0: Box, arg1: typing.SupportsInt) -> int: ...
    def n_bytes_owned(self) -> int: ...
    def n_comp(self) -> int: ...
    def num_pts(self) -> int: ...
    def resize(self, arg0: Box, arg1: typing.SupportsInt, arg2: Arena) -> None: ...
    def size(self) -> int: ...
    def small_end(self) -> IntVect2D: ...
    def to_host(self) -> BaseFab_Real: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class Box:
    big_end: IntVect2D
    hi_vect: IntVect2D
    lo_vect: IntVect2D
    small_end: IntVect2D
    def __add__(self, arg0: IntVect2D) -> Box: ...
    def __iadd__(self, arg0: IntVect2D) -> Box: ...
    @typing.overload
    def __init__(self, small: IntVect2D, big: IntVect2D) -> None: ...
    @typing.overload
    def __init__(self, small: IntVect2D, big: IntVect2D, typ: IntVect2D) -> None: ...
    @typing.overload
    def __init__(self, small: IntVect2D, big: IntVect2D, t: IndexType) -> None: ...
    @typing.overload
    def __init__(
        self,
        small: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        big: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        small: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        big: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        t: IndexType,
    ) -> None: ...
    def __isub__(self, arg0: IntVect2D) -> Box: ...
    def __iter__(self) -> collections.abc.Iterator[IntVect2D]: ...
    def __repr__(self) -> str: ...
    def __sub__(self, arg0: IntVect2D) -> Box: ...
    def begin(self, box: Box) -> Dim3: ...
    def contains(self, p: IntVect2D) -> bool:
        """
        Returns true if argument is contained within Box.
        """
    @typing.overload
    def convert(self, typ: IndexType) -> Box:
        """
        Convert the Box from the current type into the
        argument type.  This may change the Box coordinates:
        type CELL -> NODE : increase coordinate by one on high end
        type NODE -> CELL : reduce coordinate by one on high end
        other type mappings make no change.
        """
    @typing.overload
    def convert(self, typ: IntVect2D) -> Box:
        """
        Convert the Box from the current type into the
        argument type.  This may change the Box coordinates:
        type CELL -> NODE : increase coordinate by one on high end
        type NODE -> CELL : reduce coordinate by one on high end
        other type mappings make no change.
        """
    @typing.overload
    def enclosed_cells(self) -> Box:
        """
        Convert to CELL type in all directions.
        """
    @typing.overload
    def enclosed_cells(self, dir: typing.SupportsInt) -> Box:
        """
        Convert to CELL type in given direction.
        """
    @typing.overload
    def enclosed_cells(self, d: Direction) -> Box:
        """
        Convert to CELL type in given direction.
        """
    def end(self, box: Box) -> Dim3: ...
    @typing.overload
    def grow(self, n_cell: typing.SupportsInt) -> Box:
        """
        Grow Box in all directions by given amount.
        NOTE: n_cell negative shrinks the Box by that number of cells.
        """
    @typing.overload
    def grow(self, n_cells: IntVect2D) -> Box:
        """
        Grow Box in each direction by specified amount.
        """
    @typing.overload
    def grow(self, idir: typing.SupportsInt, n_cell: typing.SupportsInt) -> Box:
        """
        Grow the Box on the low and high end by n_cell cells
        in direction idir.
        """
    @typing.overload
    def grow(self, d: Direction, n_cell: typing.SupportsInt) -> Box: ...
    @typing.overload
    def grow_high(
        self, idir: typing.SupportsInt, n_cell: typing.SupportsInt = 1
    ) -> Box:
        """
        Grow the Box on the high end by n_cell cells in
        direction idir.  NOTE: n_cell negative shrinks the Box by that
        number of cells.
        """
    @typing.overload
    def grow_high(self, d: Direction, n_cell: typing.SupportsInt = 1) -> Box: ...
    @typing.overload
    def grow_low(self, idir: typing.SupportsInt, n_cell: typing.SupportsInt = 1) -> Box:
        """
        Grow the Box on the low end by n_cell cells in direction idir.
        NOTE: n_cell negative shrinks the Box by that number of cells.
        """
    @typing.overload
    def grow_low(self, d: Direction, n_cell: typing.SupportsInt = 1) -> Box: ...
    def intersects(self, b: Box) -> bool:
        """
        Returns true if Boxes have non-null intersections.
        It is an error if the Boxes have different types.
        """
    def lbound(self, arg0: Box) -> Dim3: ...
    @typing.overload
    def length(self) -> IntVect2D:
        """
        Return IntVect of lengths of the Box
        """
    @typing.overload
    def length(self, dir: typing.SupportsInt) -> int:
        """
        Return the length of the Box in given direction.
        """
    def make_slab(
        self, direction: typing.SupportsInt, slab_index: typing.SupportsInt
    ) -> Box:
        """
        Flatten the box in one direction.
        """
    def normalize(self) -> None: ...
    def numPts(self) -> int:
        """
        Return the number of points in the Box.
        """
    def same_size(self, b: Box) -> bool:
        """
        Returns true is Boxes same size, ie translates of each other,.
        It is an error if they have different types.
        """
    def same_type(self, b: Box) -> bool:
        """
        Returns true if Boxes have same type.
        """
    @typing.overload
    def shift(self, dir: typing.SupportsInt, nzones: typing.SupportsInt) -> Box:
        """
        Shift this Box nzones indexing positions in coordinate direction dir.
        """
    @typing.overload
    def shift(self, iv: IntVect2D) -> Box:
        """
        Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ...
        """
    def strictly_contains(self, p: IntVect2D) -> bool:
        """
        Returns true if argument is strictly contained within Box.
        """
    @typing.overload
    def surrounding_nodes(self) -> Box:
        """
        Convert to NODE type in all directions.
        """
    @typing.overload
    def surrounding_nodes(self, dir: typing.SupportsInt) -> Box:
        """
        Convert to NODE type in given direction.
        """
    @typing.overload
    def surrounding_nodes(self, d: Direction) -> Box:
        """
        Convert to NODE type in given direction.
        """
    def ubound(self, arg0: Box) -> Dim3: ...
    @property
    def cell_centered(self) -> bool:
        """
        Returns true if Box is cell-centered in all indexing directions.
        """
    @property
    def d_num_pts(self) -> float: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def is_square(self) -> bool: ...
    @property
    def ix_type(self) -> IndexType: ...
    @property
    def num_pts(self) -> int: ...
    @property
    def ok(self) -> bool: ...
    @property
    def size(self) -> IntVect2D: ...
    @property
    def the_unit_box() -> Box: ...
    @property
    def type(self) -> IntVect2D: ...
    @type.setter
    def type(self, arg1: IndexType) -> Box: ...
    @property
    def volume(self) -> int: ...

class BoxArray:
    def __getitem__(self, arg0: typing.SupportsInt) -> Box: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: BoxArray) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Box) -> None: ...
    def __repr__(self) -> str: ...
    def cell_equal(self, arg0: BoxArray) -> bool: ...
    def clear(self) -> None: ...
    @typing.overload
    def coarsen(self, arg0: IntVect2D) -> BoxArray: ...
    @typing.overload
    def coarsen(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @typing.overload
    def coarsenable(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> bool: ...
    @typing.overload
    def coarsenable(self, arg0: IntVect2D, arg1: typing.SupportsInt) -> bool: ...
    @typing.overload
    def coarsenable(self, arg0: IntVect2D, arg1: IntVect2D) -> bool: ...
    @typing.overload
    def convert(self, arg0: IndexType) -> BoxArray: ...
    @typing.overload
    def convert(self, arg0: IntVect2D) -> BoxArray: ...
    def define(self, arg0: Box) -> None: ...
    @typing.overload
    def enclosed_cells(self) -> BoxArray: ...
    @typing.overload
    def enclosed_cells(self, arg0: typing.SupportsInt) -> BoxArray: ...
    def get(self, arg0: typing.SupportsInt) -> Box: ...
    def ix_type(self) -> IndexType: ...
    @typing.overload
    def max_size(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @typing.overload
    def max_size(self, arg0: IntVect2D) -> BoxArray: ...
    def minimal_box(self) -> Box: ...
    @typing.overload
    def refine(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @typing.overload
    def refine(self, arg0: IntVect2D) -> BoxArray: ...
    def resize(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def surroundingNodes(self) -> BoxArray: ...
    @typing.overload
    def surroundingNodes(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @property
    def capacity(self) -> int: ...
    @property
    def d_numPts(self) -> float: ...
    @property
    def empty(self) -> bool: ...
    @property
    def numPts(self) -> int: ...
    @property
    def size(self) -> int: ...

class Config:
    amrex_version: typing.ClassVar[str] = "25.11-15-g4dad1664d746"
    gpu_backend = None
    have_eb: typing.ClassVar[bool] = True
    have_gpu: typing.ClassVar[bool] = False
    have_mpi: typing.ClassVar[bool] = True
    have_omp: typing.ClassVar[bool] = False
    have_simd: typing.ClassVar[bool] = False
    precision: typing.ClassVar[str] = "DOUBLE"
    precision_particles: typing.ClassVar[str] = "DOUBLE"
    simd_size: typing.ClassVar[int] = 1
    spacedim: typing.ClassVar[int] = 2
    verbose: typing.ClassVar[int] = 1

class CoordSys:
    class CoordType(enum.IntEnum):
        """
        An enumeration.
        """

        RZ: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.RZ: 1>
        SPHERICAL: typing.ClassVar[
            CoordSys.CoordType
        ]  # value = <CoordType.SPHERICAL: 2>
        cartesian: typing.ClassVar[
            CoordSys.CoordType
        ]  # value = <CoordType.cartesian: 0>
        undef: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.undef: -1>

    RZ: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.RZ: 1>
    SPHERICAL: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.SPHERICAL: 2>
    cartesian: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.cartesian: 0>
    undef: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.undef: -1>
    def Coord(self) -> CoordSys.CoordType: ...
    def CoordInt(self) -> int: ...
    def IsCartesian(self) -> bool: ...
    def IsRZ(self) -> bool: ...
    def IsSPHERICAL(self) -> bool: ...
    def SetCoord(self, arg0: CoordSys.CoordType) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: CoordSys) -> None: ...
    def __repr__(self) -> str: ...
    def ok(self) -> bool: ...

class Dim3:
    def __init__(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def x(self) -> int: ...
    @x.setter
    def x(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def y(self) -> int: ...
    @y.setter
    def y(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def z(self) -> int: ...
    @z.setter
    def z(self, arg0: typing.SupportsInt) -> None: ...

class Direction:
    pass

class DistributionMapping:
    def ProcessorMap(self) -> Vector_int: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: DistributionMapping) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None: ...
    @typing.overload
    def __init__(self, boxes: BoxArray) -> None: ...
    @typing.overload
    def __init__(self, boxes: BoxArray, nprocs: typing.SupportsInt) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def define(self, boxes: BoxArray) -> None: ...
    @typing.overload
    def define(self, boxes: BoxArray, nprocs: typing.SupportsInt) -> None: ...
    @typing.overload
    def define(self, arg0: Vector_int) -> None: ...
    @property
    def capacity(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def link_count(self) -> int: ...
    @property
    def size(self) -> int: ...

class EBFArrayBoxFactory(FabFactory_FArrayBox):
    def getVolFrac(self) -> MultiFab:
        """
        Return volume faction MultiFab
        """

class EBSupport(enum.Enum):
    """
    An enumeration.
    """

    basic: typing.ClassVar[EBSupport]  # value = <EBSupport.basic: 1>
    full: typing.ClassVar[EBSupport]  # value = <EBSupport.full: 3>
    volume: typing.ClassVar[EBSupport]  # value = <EBSupport.volume: 2>

class FArrayBox(BaseFab_Real):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Arena) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: typing.SupportsInt, arg2: Arena) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: typing.SupportsInt, arg2: bool, arg3: bool, arg4: Arena
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: typing.SupportsInt, arg2: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: typing.SupportsInt, arg2: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double, arg1: IndexType) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const, arg1: IndexType) -> None: ...
    def __repr__(self) -> str: ...

class FabArrayBase:
    @staticmethod
    def __iter__(fab): ...
    def __len__(self) -> int:
        """
        Return the number of FABs in the FabArray.
        """
    def is_nodal(self, arg0: typing.SupportsInt) -> bool: ...
    @property
    def is_all_cell_centered(self) -> bool: ...
    @property
    def is_all_nodal(self) -> bool: ...
    @property
    def nComp(self) -> int:
        """
        Return number of variables (aka components) associated with each point.
        """
    @property
    def n_grow_vect(self) -> IntVect2D:
        """
        Return the grow factor (per direction) that defines the region of definition.
        """
    @property
    def num_comp(self) -> int:
        """
        Return number of variables (aka components) associated with each point.
        """
    @property
    def size(self) -> int:
        """
        Return the number of FABs in the FabArray.
        """

class FabArray_FArrayBox(FabArrayBase):
    @typing.overload
    def abs(
        self,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def abs(
        self, comp: typing.SupportsInt, ncomp: typing.SupportsInt, nghost: IntVect2D
    ) -> None: ...
    def array(self, arg0: MFIter) -> Array4_double: ...
    def clear(self) -> None: ...
    def const_array(self, arg0: MFIter) -> Array4_double_const: ...
    @typing.overload
    def fill_boundary(self, cross: bool = False) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(self, period: Periodicity, cross: bool = False) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self, nghost: IntVect2D, period: Periodicity, cross: bool = False
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self, scomp: typing.SupportsInt, ncomp: typing.SupportsInt, cross: bool = False
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        period: Periodicity,
        cross: bool = False,
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        period: Periodicity,
        cross: bool = False,
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    def lin_comb(
        self,
        a: typing.SupportsFloat,
        x: FabArray_FArrayBox,
        xcomp: typing.SupportsInt,
        b: typing.SupportsFloat,
        y: FabArray_FArrayBox,
        ycomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self = a * x + b * y

        Parameters
        ----------
        a     : float
            scalar a
        x     : FabArray
        xcomp : int
            starting component of x
        b     : float
            scalar b
        y     : FabArray
        ycomp : int
            starting component of y
        comp  : int
            starting component of self
        numcomp : int
            number of components
        nghost  : int
            number of ghost cells
        """
    def ok(self) -> bool: ...
    @typing.overload
    def override_sync(self, period: Periodicity) -> None:
        """
        Synchronize nodal data.

            The synchronization will override valid regions by the intersecting valid regions with a higher precedence.
            The smaller the global box index is, the higher precedence the box has.
            With periodic boundaries, for cells in the same box, those near the lower corner have higher precedence than those near the upper corner.

            Parameters
            ----------
            scomp :
              starting component
            ncomp :
              number of components
            period :
              periodic length if it's non-zero
        """
    @typing.overload
    def override_sync(
        self, scomp: typing.SupportsInt, ncomp: typing.SupportsInt, period: Periodicity
    ) -> None:
        """
        Synchronize nodal data.

            The synchronization will override valid regions by the intersecting valid regions with a higher precedence.
            The smaller the global box index is, the higher precedence the box has.
            With periodic boundaries, for cells in the same box, those near the lower corner have higher precedence than those near the upper corner.

            Parameters
            ----------
            scomp :
              starting component
            ncomp :
              number of components
            period :
              periodic length if it's non-zero
        """
    def saxpy(
        self,
        a: typing.SupportsFloat,
        x: FabArray_FArrayBox,
        x_comp: typing.SupportsInt,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self += a * x

        Parameters
        ----------
        a      : scalar a
        x      : FabArray x
        x_comp : starting component of x
        comp   : starting component of self
        ncomp  : number of components
        nghost : number of ghost cells
        """
    @typing.overload
    def set_val(self, val: typing.SupportsFloat) -> None:
        """
        Set all components in the entire region of each FAB to val.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsFloat,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsFloat,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsFloat,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsFloat,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """
    def sum(self, comp: typing.SupportsInt, nghost: IntVect2D, local: bool) -> float:
        """
        Returns the sum of component "comp"
        """
    @typing.overload
    def sum_boundary(self, period: Periodicity, deterministic: bool = False) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        dst_nghost: IntVect2D,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    def xpay(
        self,
        a: typing.SupportsFloat,
        x: FabArray_FArrayBox,
        xcomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self = x + a * self

        Parameters
        ----------
        a      : scalar a
        x      : FabArray x
        x_comp : starting component of x
        comp   : starting component of self
        ncomp  : number of components
        nghost : number of ghost cells
        """
    @property
    def arena(self) -> Arena:
        """
        Provides access to the Arena this FabArray was build with.
        """
    @property
    def factory(self) -> FabFactory_FArrayBox: ...
    @property
    def has_EB_fab_factory(self) -> bool: ...

class FabArray_IArrayBox(FabArrayBase):
    @typing.overload
    def abs(
        self,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def abs(
        self, comp: typing.SupportsInt, ncomp: typing.SupportsInt, nghost: IntVect2D
    ) -> None: ...
    def array(self, arg0: MFIter) -> Array4_int: ...
    def clear(self) -> None: ...
    def const_array(self, arg0: MFIter) -> Array4_int_const: ...
    @typing.overload
    def fill_boundary(self, cross: bool = False) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(self, period: Periodicity, cross: bool = False) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self, nghost: IntVect2D, period: Periodicity, cross: bool = False
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self, scomp: typing.SupportsInt, ncomp: typing.SupportsInt, cross: bool = False
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        period: Periodicity,
        cross: bool = False,
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    @typing.overload
    def fill_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        period: Periodicity,
        cross: bool = False,
    ) -> None:
        """
        Copy on intersection within a FabArray.

            Data is copied from valid regions to intersecting regions of definition.
            The purpose is to fill in the boundary regions of each FAB in the FabArray.
            If cross=true, corner cells are not filled. If the length of periodic is provided,
            periodic boundaries are also filled.

            If scomp is provided, this only copies ncomp components starting at scomp.

            Note that FabArray itself does not contains any periodicity information.
            FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.
        """
    def lin_comb(
        self,
        a: typing.SupportsInt,
        x: FabArray_IArrayBox,
        xcomp: typing.SupportsInt,
        b: typing.SupportsInt,
        y: FabArray_IArrayBox,
        ycomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self = a * x + b * y

        Parameters
        ----------
        a     : float
            scalar a
        x     : FabArray
        xcomp : int
            starting component of x
        b     : float
            scalar b
        y     : FabArray
        ycomp : int
            starting component of y
        comp  : int
            starting component of self
        numcomp : int
            number of components
        nghost  : int
            number of ghost cells
        """
    def ok(self) -> bool: ...
    @typing.overload
    def override_sync(self, period: Periodicity) -> None:
        """
        Synchronize nodal data.

            The synchronization will override valid regions by the intersecting valid regions with a higher precedence.
            The smaller the global box index is, the higher precedence the box has.
            With periodic boundaries, for cells in the same box, those near the lower corner have higher precedence than those near the upper corner.

            Parameters
            ----------
            scomp :
              starting component
            ncomp :
              number of components
            period :
              periodic length if it's non-zero
        """
    @typing.overload
    def override_sync(
        self, scomp: typing.SupportsInt, ncomp: typing.SupportsInt, period: Periodicity
    ) -> None:
        """
        Synchronize nodal data.

            The synchronization will override valid regions by the intersecting valid regions with a higher precedence.
            The smaller the global box index is, the higher precedence the box has.
            With periodic boundaries, for cells in the same box, those near the lower corner have higher precedence than those near the upper corner.

            Parameters
            ----------
            scomp :
              starting component
            ncomp :
              number of components
            period :
              periodic length if it's non-zero
        """
    def saxpy(
        self,
        a: typing.SupportsInt,
        x: FabArray_IArrayBox,
        x_comp: typing.SupportsInt,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self += a * x

        Parameters
        ----------
        a      : scalar a
        x      : FabArray x
        x_comp : starting component of x
        comp   : starting component of self
        ncomp  : number of components
        nghost : number of ghost cells
        """
    @typing.overload
    def set_val(self, val: typing.SupportsInt) -> None:
        """
        Set all components in the entire region of each FAB to val.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsInt,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsInt,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsInt,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """
    @typing.overload
    def set_val(
        self,
        val: typing.SupportsInt,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """
    def sum(self, comp: typing.SupportsInt, nghost: IntVect2D, local: bool) -> int:
        """
        Returns the sum of component "comp"
        """
    @typing.overload
    def sum_boundary(self, period: Periodicity, deterministic: bool = False) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    @typing.overload
    def sum_boundary(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
        dst_nghost: IntVect2D,
        period: Periodicity,
        deterministic: bool = False,
    ) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """
    def xpay(
        self,
        a: typing.SupportsInt,
        x: FabArray_IArrayBox,
        xcomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self = x + a * self

        Parameters
        ----------
        a      : scalar a
        x      : FabArray x
        x_comp : starting component of x
        comp   : starting component of self
        ncomp  : number of components
        nghost : number of ghost cells
        """
    @property
    def arena(self) -> Arena:
        """
        Provides access to the Arena this FabArray was build with.
        """
    @property
    def factory(self) -> FabFactory_IArrayBox: ...
    @property
    def has_EB_fab_factory(self) -> bool: ...

class FabFactory_FArrayBox:
    pass

class FabFactory_IArrayBox:
    pass

class Geometry(CoordSys):
    @typing.overload
    def ProbHi(self, dir: typing.SupportsInt) -> float:
        """
        Get the hi end of the problem domain in specified direction
        """
    @typing.overload
    def ProbHi(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Get the list of lo ends of the problem domain
        """
    def ProbLength(self, arg0: typing.SupportsInt) -> float:
        """
        length of problem domain in specified dimension
        """
    @typing.overload
    def ProbLo(self, dir: typing.SupportsInt) -> float:
        """
        Get the lo end of the problem domain in specified direction
        """
    @typing.overload
    def ProbLo(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Get the list of lo ends of the problem domain
        """
    def ProbSize(self) -> float:
        """
        the overall size of the domain
        """
    def ResetDefaultCoord(self: typing.SupportsInt) -> None:
        """
        Reset default coord of Geometry class with an Array of `int`
        """
    def ResetDefaultPeriodicity(
        self: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None:
        """
        Reset default periodicity of Geometry class with an Array of `int`
        """
    def ResetDefaultProbDomain(self: RealBox) -> None:
        """
        Reset default problem domain of Geometry class with a `RealBox`
        """
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        dom: Box,
        rb: RealBox,
        coord: typing.SupportsInt,
        is_per: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def coarsen(self, rr: IntVect2D) -> None: ...
    def data(self) -> GeometryData:
        """
        Returns non-static copy of geometry's stored data
        """
    def define(
        self,
        dom: Box,
        rb: RealBox,
        coord: typing.SupportsInt,
        is_per: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None:
        """
        Set geometry
        """
    @typing.overload
    def growNonPeriodicDomain(self, ngrow: IntVect2D) -> Box: ...
    @typing.overload
    def growNonPeriodicDomain(self, ngrow: typing.SupportsInt) -> Box: ...
    @typing.overload
    def growPeriodicDomain(self, ngrow: IntVect2D) -> Box: ...
    @typing.overload
    def growPeriodicDomain(self, ngrow: typing.SupportsInt) -> Box: ...
    def insideRoundOffDomain(
        self, x: typing.SupportsFloat, y: typing.SupportsFloat
    ) -> bool:
        """
        Returns true if a point is inside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()
        """
    def isAllPeriodic(self) -> bool:
        """
        Is domain periodic in all directions?
        """
    def isAnyPeriodic(self) -> bool:
        """
        Is domain periodic in any direction?
        """
    @typing.overload
    def isPeriodic(self, arg0: typing.SupportsInt) -> bool:
        """
        Is the domain periodic in the specified direction?
        """
    @typing.overload
    def isPeriodic(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        """
        Return list indicating whether domain is periodic in each direction
        """
    def outsideRoundOffDomain(
        self, x: typing.SupportsFloat, y: typing.SupportsFloat
    ) -> bool:
        """
        Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()
        """
    def period(self, dir: typing.SupportsInt) -> int:
        """
        Return the period in the specified direction
        """
    @typing.overload
    def periodicity(self) -> Periodicity: ...
    @typing.overload
    def periodicity(self, b: Box) -> Periodicity:
        """
        Return Periodicity object with lengths determined by input Box
        """
    def refine(self, rr: IntVect2D) -> None: ...
    def setPeriodicity(
        self,
        period: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> typing.Annotated[list[int], "FixedSize(2)"]:
        """
        Set periodicity flags and return the old flags.
        Note that, unlike Periodicity class, the flags are just boolean.
        """
    @property
    def domain(self) -> Box:
        """
        The rectangular domain (index space).
        """
    @domain.setter
    def domain(self, arg1: Box) -> None: ...
    @property
    def prob_domain(self) -> RealBox:
        """
        The problem domain (real).
        """
    @prob_domain.setter
    def prob_domain(self, arg1: RealBox) -> None: ...

class GeometryData:
    @typing.overload
    def CellSize(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Returns the cellsize for each coordinate direction.
        """
    @typing.overload
    def CellSize(self, arg0: typing.SupportsInt) -> float:
        """
        Returns the cellsize for specified coordinate direction.
        """
    def Coord(self) -> int:
        """
        return integer coordinate type
        """
    def Domain(self) -> Box:
        """
        Returns our rectangular domain
        """
    @typing.overload
    def ProbHi(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Returns the hi end for each coordinate direction.
        """
    @typing.overload
    def ProbHi(self, arg0: typing.SupportsInt) -> float:
        """
        Returns the hi end of the problem domain in specified dimension.
        """
    @typing.overload
    def ProbLo(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Returns the lo end for each coordinate direction.
        """
    @typing.overload
    def ProbLo(self, arg0: typing.SupportsInt) -> float:
        """
        Returns the lo end of the problem domain in specified dimension.
        """
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def isPeriodic(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        """
        Returns whether the domain is periodic in each direction.
        """
    @typing.overload
    def isPeriodic(self, arg0: typing.SupportsInt) -> int:
        """
        Returns whether the domain is periodic in the given direction.
        """
    @property
    def coord(self) -> int:
        """
        The Coordinates type.
        """
    @property
    def domain(self) -> Box:
        """
        The index domain.
        """
    @property
    def dx(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        The cellsize for each coordinate direction.
        """
    @property
    def is_periodic(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        """
        Returns whether the domain is periodic in each coordinate direction.
        """
    @property
    def prob_domain(self) -> RealBox:
        """
        The problem domain (real).
        """

class GrowthStrategy(enum.Enum):
    """
    An enumeration.
    """

    Exact: typing.ClassVar[GrowthStrategy]  # value = <GrowthStrategy.Exact: 1>
    Geometric: typing.ClassVar[GrowthStrategy]  # value = <GrowthStrategy.Geometric: 2>
    Poisson: typing.ClassVar[GrowthStrategy]  # value = <GrowthStrategy.Poisson: 0>

class IndexType:
    class CellIndex(enum.IntEnum):
        """
        An enumeration.
        """

        CELL: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.CELL: 0>
        NODE: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.NODE: 1>

    CELL: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.CELL: 0>
    NODE: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.NODE: 1>
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_type() -> IndexType: ...
    @staticmethod
    def node_type() -> IndexType: ...
    def __eq__(self, arg0: IndexType) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: IndexType) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: IndexType.CellIndex, arg1: IndexType.CellIndex
    ) -> None: ...
    def __len__(self) -> int: ...
    def __lt__(self, arg0: IndexType) -> bool: ...
    def __ne__(self, arg0: IndexType) -> bool: ...
    def __repr__(self) -> str: ...
    def __str(self) -> str: ...
    def any(self) -> bool: ...
    @typing.overload
    def cell_centered(self) -> bool: ...
    @typing.overload
    def cell_centered(self, arg0: typing.SupportsInt) -> bool: ...
    def clear(self) -> None: ...
    def flip(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def ix_type(self) -> IntVect2D: ...
    @typing.overload
    def ix_type(self, arg0: typing.SupportsInt) -> IndexType.CellIndex: ...
    @typing.overload
    def node_centered(self) -> bool: ...
    @typing.overload
    def node_centered(self, arg0: typing.SupportsInt) -> bool: ...
    def ok(self) -> bool: ...
    def set(self, arg0: typing.SupportsInt) -> None: ...
    def set_type(self, arg0: typing.SupportsInt, arg1: IndexType.CellIndex) -> None: ...
    def setall(self) -> None: ...
    def test(self, arg0: typing.SupportsInt) -> bool: ...
    def to_IntVect(self) -> IntVect2D: ...
    def unset(self, arg0: typing.SupportsInt) -> None: ...

class IntVect1D:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_vector() -> IntVect1D: ...
    @staticmethod
    def max_vector() -> IntVect1D: ...
    @staticmethod
    def min_vector() -> IntVect1D: ...
    @staticmethod
    def node_vector() -> IntVect1D: ...
    @staticmethod
    def unit_vector() -> IntVect1D: ...
    @staticmethod
    def zero_vector() -> IntVect1D: ...
    @typing.overload
    def __add__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __add__(self, arg0: IntVect1D) -> IntVect1D: ...
    @typing.overload
    def __eq__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __eq__(self, arg0: IntVect1D) -> bool: ...
    def __ge__(self, arg0: IntVect1D) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    def __gt__(self, arg0: IntVect1D) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __iadd__(self, arg0: IntVect1D) -> IntVect1D: ...
    @typing.overload
    def __imul__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __imul__(self, arg0: IntVect1D) -> IntVect1D: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @typing.overload
    def __isub__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __isub__(self, arg0: IntVect1D) -> IntVect1D: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    @typing.overload
    def __itruediv__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __itruediv__(self, arg0: IntVect1D) -> IntVect1D: ...
    def __le__(self, arg0: IntVect1D) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, arg0: IntVect1D) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __mul__(self, arg0: IntVect1D) -> IntVect1D: ...
    @typing.overload
    def __ne__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __ne__(self, arg0: IntVect1D) -> bool: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> int: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __sub__(self, arg0: IntVect1D) -> IntVect1D: ...
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsInt) -> IntVect1D: ...
    @typing.overload
    def __truediv__(self, arg0: IntVect1D) -> IntVect1D: ...
    def dim3(self) -> Dim3: ...
    def numpy(self) -> numpy.ndarray: ...
    @property
    def max(self) -> int: ...
    @property
    def min(self) -> int: ...
    @property
    def sum(self) -> int: ...

class IntVect2D:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_vector() -> IntVect2D: ...
    @staticmethod
    def max_vector() -> IntVect2D: ...
    @staticmethod
    def min_vector() -> IntVect2D: ...
    @staticmethod
    def node_vector() -> IntVect2D: ...
    @staticmethod
    def unit_vector() -> IntVect2D: ...
    @staticmethod
    def zero_vector() -> IntVect2D: ...
    @typing.overload
    def __add__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __add__(self, arg0: IntVect2D) -> IntVect2D: ...
    @typing.overload
    def __eq__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __eq__(self, arg0: IntVect2D) -> bool: ...
    def __ge__(self, arg0: IntVect2D) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    def __gt__(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __iadd__(self, arg0: IntVect2D) -> IntVect2D: ...
    @typing.overload
    def __imul__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __imul__(self, arg0: IntVect2D) -> IntVect2D: ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def __isub__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __isub__(self, arg0: IntVect2D) -> IntVect2D: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    @typing.overload
    def __itruediv__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __itruediv__(self, arg0: IntVect2D) -> IntVect2D: ...
    def __le__(self, arg0: IntVect2D) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, arg0: IntVect2D) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __mul__(self, arg0: IntVect2D) -> IntVect2D: ...
    @typing.overload
    def __ne__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __ne__(self, arg0: IntVect2D) -> bool: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> int: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __sub__(self, arg0: IntVect2D) -> IntVect2D: ...
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __truediv__(self, arg0: IntVect2D) -> IntVect2D: ...
    def dim3(self) -> Dim3: ...
    def numpy(self) -> numpy.ndarray: ...
    @property
    def max(self) -> int: ...
    @property
    def min(self) -> int: ...
    @property
    def sum(self) -> int: ...

class IntVect3D:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_vector() -> IntVect3D: ...
    @staticmethod
    def max_vector() -> IntVect3D: ...
    @staticmethod
    def min_vector() -> IntVect3D: ...
    @staticmethod
    def node_vector() -> IntVect3D: ...
    @staticmethod
    def unit_vector() -> IntVect3D: ...
    @staticmethod
    def zero_vector() -> IntVect3D: ...
    @typing.overload
    def __add__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __add__(self, arg0: IntVect3D) -> IntVect3D: ...
    @typing.overload
    def __eq__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __eq__(self, arg0: IntVect3D) -> bool: ...
    def __ge__(self, arg0: IntVect3D) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    def __gt__(self, arg0: IntVect3D) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __iadd__(self, arg0: IntVect3D) -> IntVect3D: ...
    @typing.overload
    def __imul__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __imul__(self, arg0: IntVect3D) -> IntVect3D: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"
        ],
    ) -> None: ...
    @typing.overload
    def __isub__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __isub__(self, arg0: IntVect3D) -> IntVect3D: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    @typing.overload
    def __itruediv__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __itruediv__(self, arg0: IntVect3D) -> IntVect3D: ...
    def __le__(self, arg0: IntVect3D) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, arg0: IntVect3D) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __mul__(self, arg0: IntVect3D) -> IntVect3D: ...
    @typing.overload
    def __ne__(self, arg0: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __ne__(self, arg0: IntVect3D) -> bool: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> int: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __sub__(self, arg0: IntVect3D) -> IntVect3D: ...
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsInt) -> IntVect3D: ...
    @typing.overload
    def __truediv__(self, arg0: IntVect3D) -> IntVect3D: ...
    def dim3(self) -> Dim3: ...
    def numpy(self) -> numpy.ndarray: ...
    @property
    def max(self) -> int: ...
    @property
    def min(self) -> int: ...
    @property
    def sum(self) -> int: ...

class MFInfo:
    alloc: bool
    arena: Arena
    tags: Vector_string
    def __init__(self) -> None: ...
    def set_alloc(self, arg0: bool) -> MFInfo: ...
    def set_arena(self, arg0: Arena) -> MFInfo: ...
    def set_tag(self, arg0: str) -> None: ...

class MFItInfo:
    device_sync: bool
    do_tiling: bool
    dynamic: bool
    tilesize: IntVect2D
    def __init__(self) -> None: ...
    def disable_device_sync(self) -> MFItInfo: ...
    def enable_tiling(self, ts: IntVect2D) -> MFItInfo: ...
    def set_device_sync(self, f: bool) -> MFItInfo: ...
    def set_dynamic(self, f: bool) -> MFItInfo: ...
    def set_num_streams(self, n: typing.SupportsInt) -> MFItInfo: ...
    def use_default_stream(self) -> MFItInfo: ...
    @property
    def num_streams(self) -> int: ...
    @num_streams.setter
    def num_streams(self, arg0: typing.SupportsInt) -> None: ...

class MFIter:
    @typing.overload
    def __init__(self, arg0: FabArrayBase) -> None: ...
    @typing.overload
    def __init__(self, arg0: FabArrayBase, arg1: MFItInfo) -> None: ...
    @typing.overload
    def __init__(self, arg0: MultiFab) -> None: ...
    @typing.overload
    def __init__(self, arg0: MultiFab, arg1: MFItInfo) -> None: ...
    @typing.overload
    def __init__(self, arg0: iMultiFab) -> None: ...
    @typing.overload
    def __init__(self, arg0: iMultiFab, arg1: MFItInfo) -> None: ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...
    def _incr(self) -> None: ...
    def fabbox(self) -> Box: ...
    def finalize(self) -> None: ...
    @typing.overload
    def grownnodaltilebox(
        self, int: typing.SupportsInt = -1, ng: typing.SupportsInt = -1000000
    ) -> Box: ...
    @typing.overload
    def grownnodaltilebox(self, int: typing.SupportsInt, ng: IntVect2D) -> Box: ...
    def growntilebox(self, ng: IntVect2D = -1000000) -> Box: ...
    def nodaltilebox(self, dir: typing.SupportsInt = -1) -> Box: ...
    @typing.overload
    def tilebox(self) -> Box: ...
    @typing.overload
    def tilebox(self, arg0: IntVect2D) -> Box: ...
    @typing.overload
    def tilebox(self, arg0: IntVect2D, arg1: IntVect2D) -> Box: ...
    def validbox(self) -> Box: ...
    @property
    def index(self) -> int: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def length(self) -> int: ...

class MPMD_Copier:
    @typing.overload
    def __init__(self, arg0: bool) -> None: ...
    @typing.overload
    def __init__(
        self, ba: BoxArray, dm: DistributionMapping, send_ba: bool = False
    ) -> None: ...
    def box_array(self) -> BoxArray: ...
    def distribution_map(self) -> DistributionMapping: ...
    def recv(
        self,
        arg0: FabArray_FArrayBox,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    def send(
        self,
        arg0: FabArray_FArrayBox,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...

class MultiFab(FabArray_FArrayBox):
    @staticmethod
    def __iter__(mfab): ...
    @staticmethod
    def finalize() -> None: ...
    @staticmethod
    def initialize() -> None: ...
    def __getitem__(self, index, with_internal_ghosts=False):
        """
        Returns slice of the MultiFab using global indexing, as a numpy array.
            This uses numpy array indexing, with the indexing relative to the global array.
            The slice ranges can cross multiple blocks and the result will be gathered into a single
            array.

            In an MPI context, this is a global operation. An "allgather" is performed so that the full
            result is returned on all processors.

            Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

            The default range of the indices includes only the valid cells. The ":" index will include all of
            the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
            negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
            The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
            ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

            Parameters
            ----------
            index : the index using numpy style indexing
                Index of the slice to return.
            with_internal_ghosts : bool, optional
                Whether to include internal ghost cells. When true, data from ghost cells may be used that
                overlaps valid cells.

        """
    @typing.overload
    def __init__(self) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions
                    inherited from FabArray.
        """
    @typing.overload
    def __init__(self, a: Arena) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions.
                    If ``define`` is called later with a nullptr as MFInfo's arena, the
                    default Arena ``a`` will be used.  If the arena in MFInfo is not a
                    nullptr, the MFInfo's arena will be used.
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
        info: MFInfo,
        factory: FabFactory_FArrayBox,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
        info: MFInfo,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        info: MFInfo,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        info: MFInfo,
        factory: FabFactory_FArrayBox,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    def __repr__(self) -> str: ...
    def __setitem__(self, index, value):
        """
        Sets the slice of the MultiFab using global indexing.
            This uses numpy array indexing, with the indexing relative to the global array.
            The slice ranges can cross multiple blocks and the value will be distributed accordingly.
            Note that this will apply the value to both valid and ghost cells.

            In an MPI context, this is a local operation. On each processor, the blocks within the slice
            range will be set to the value.

            Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

            The default range of the indices includes only the valid cells. The ":" index will include all of
            the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
            negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
            The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
            ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

            Parameters
            ----------
            index : the index using numpy style indexing
                Index of the slice to return.
            value : scalar or array
                Input value to assign to the specified slice of the MultiFab

        """
    @typing.overload
    def add(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def add(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def add_product(
        self,
        src1: MultiFab,
        comp1: typing.SupportsInt,
        src2: MultiFab,
        comp2: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        self += src1 * src2
        """
    @typing.overload
    def add_product(
        self,
        src1: MultiFab,
        comp1: typing.SupportsInt,
        src2: MultiFab,
        comp2: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        self += src1 * src2
        """
    def average_sync(self, arg0: Periodicity) -> None: ...
    def box_array(self: FabArrayBase) -> BoxArray: ...
    @typing.overload
    def contains_inf(self, local: bool = False) -> bool: ...
    @typing.overload
    def contains_inf(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
        local: bool = False,
    ) -> bool: ...
    @typing.overload
    def contains_inf(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        local: bool = False,
    ) -> bool: ...
    @typing.overload
    def contains_nan(self, local: bool = False) -> bool: ...
    @typing.overload
    def contains_nan(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
        local: bool = False,
    ) -> bool: ...
    @typing.overload
    def contains_nan(
        self,
        scomp: typing.SupportsInt,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        local: bool = False,
    ) -> bool: ...
    def copy(self):
        """

        Create a copy of this MultiFab, using the same Arena.

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX

        Returns
        -------
        amrex.MultiFab
            A copy of this MultiFab.

        """
    @typing.overload
    def copymf(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        dstcomp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Copy from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray. The copy is local
        """
    @typing.overload
    def copymf(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        dstcomp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Copy from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray. The copy is local
        """
    def divi(
        self,
        mf: MultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function divides the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The division is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.  Note, nothing is done to protect
        against divide by zero.
        """
    @typing.overload
    def divide(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def divide(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    def dm(self: FabArrayBase) -> DistributionMapping: ...
    @typing.overload
    def dot(
        self,
        comp: typing.SupportsInt,
        y: MultiFab,
        y_comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
        local: bool = False,
    ) -> float:
        """
        Returns the dot product of self with another MultiFab.
        """
    @typing.overload
    def dot(
        self,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
        local: bool = False,
    ) -> float:
        """
        Returns the dot product with itself.
        """
    @typing.overload
    def dot(
        self,
        mask: iMultiFab,
        comp: typing.SupportsInt,
        y: MultiFab,
        y_comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
        local: bool = False,
    ) -> float:
        """
        Returns the dot product of self with another MultiFab where the mask is valid.
        """
    def imesh(self, idir, include_ghosts=False):
        """
        Returns the integer mesh along the specified direction with the appropriate centering.
            This is the location of the data points in grid cell units.

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX
            direction : integer
                Zero based direction number.
                In a typical Cartesian case, 0 would be 'x' direction.
            include_ghosts : bool, default=False
                Whether or not ghost cells are included in the mesh.

        """
    @typing.overload
    def invert(
        self, numerator: typing.SupportsFloat, nghost: typing.SupportsInt
    ) -> None:
        """
        Replaces the value of each cell in the specified subregion of
        the MultiFab with its reciprocal multiplied by the value of
        numerator.  The value of nghost specifies the number of cells
        in the boundary region that should be modified.
        """
    @typing.overload
    def invert(
        self,
        numerator: typing.SupportsFloat,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Replaces the value of each cell in the specified subregion of
        the MultiFab with its reciprocal multiplied by the value of
        numerator. The subregion consists of the num_comp components
        starting at component comp.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in the
        subregion that should be modified.
        """
    @typing.overload
    def invert(
        self, numerator: typing.SupportsFloat, region: Box, nghost: typing.SupportsInt
    ) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """
    @typing.overload
    def invert(
        self,
        numerator: typing.SupportsFloat,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of invert(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in the
        subregion that should be modified.
        """
    def lin_comb(
        self,
        a: typing.SupportsFloat,
        x: MultiFab,
        x_comp: typing.SupportsInt,
        b: typing.SupportsFloat,
        y: MultiFab,
        y_comp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        self = a * x + b * y
        """
    @typing.overload
    def max(
        self,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> float:
        """
        Returns the maximum value of the specified component of the (i)MultiFab.
        """
    @typing.overload
    def max(
        self,
        region: Box,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> float:
        """
        Returns the maximum value of the specified component of the (i)MultiFab over the region.
        """
    def maxIndex(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> IntVect2D: ...
    @typing.overload
    def min(
        self,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> float:
        """
        Returns the minimum value of the specified component of the (i)MultiFab.
        """
    @typing.overload
    def min(
        self,
        region: Box,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> float:
        """
        Returns the minimum value of the specified component of the (i)MultiFab over the region.
        """
    def minIndex(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> IntVect2D: ...
    def minus(
        self,
        mf: MultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function subtracts the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The subtraction is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.
        """
    @typing.overload
    def mult(self, val: typing.SupportsFloat, nghost: typing.SupportsInt = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val).
        The value of nghost specifies the number of cells in the
        boundary region that should be modified.
        """
    @typing.overload
    def mult(
        self,
        val: typing.SupportsFloat,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Scales the value of each cell in the specified subregion of the
        MultiFab by the scalar val (a[i] <- a[i]*val). The subregion
        consists of the num_comp components starting at component comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """
    @typing.overload
    def mult(
        self,
        val: typing.SupportsFloat,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of mult(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """
    @typing.overload
    def mult(
        self, val: typing.SupportsFloat, region: Box, nghost: typing.SupportsInt = 0
    ) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """
    @typing.overload
    def multiply(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def multiply(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def negate(self, nghost: typing.SupportsInt = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab.  The value of nghost specifies the number of
        cells in the boundary region that should be modified.
        """
    @typing.overload
    def negate(
        self,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Negates the value of each cell in the specified subregion of
        the MultiFab.  The subregion consists of the num_comp
        components starting at component comp.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """
    @typing.overload
    def negate(self, region: Box, nghost: typing.SupportsInt = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab that also intersects the Box region.  The value
        of nghost specifies the number of cells in the boundary region
        that should be modified.
        """
    @typing.overload
    def negate(
        self,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of negate(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """
    @typing.overload
    def norm0(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: bool, arg3: bool
    ) -> float: ...
    @typing.overload
    def norm0(
        self,
        arg0: iMultiFab,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
        arg3: bool,
    ) -> float: ...
    @typing.overload
    def norm1(
        self, arg0: typing.SupportsInt, arg1: Periodicity, arg2: bool
    ) -> float: ...
    @typing.overload
    def norm1(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: bool
    ) -> float: ...
    @typing.overload
    def norm1(
        self, arg0: Vector_int, arg1: typing.SupportsInt, arg2: bool
    ) -> Vector_Real: ...
    @typing.overload
    def norm2(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def norm2(self, arg0: typing.SupportsInt, arg1: Periodicity) -> float: ...
    @typing.overload
    def norm2(self, arg0: Vector_int) -> Vector_Real: ...
    def norminf(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: bool, arg3: bool
    ) -> float: ...
    def override_sync(self, arg0: iMultiFab, arg1: Periodicity) -> None: ...
    @typing.overload
    def plus(self, val: typing.SupportsFloat, nghost: typing.SupportsInt = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab.  The value
        of nghost specifies the number of cells in the boundary
        region that should be modified.
        """
    @typing.overload
    def plus(
        self,
        val: typing.SupportsFloat,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Adds the scalar value \\p val to the value of each cell in the
        specified subregion of the MultiFab.

        The subregion consists of the \\p num_comp components starting at component \\p comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """
    @typing.overload
    def plus(
        self, val: typing.SupportsFloat, region: Box, nghost: typing.SupportsInt = 0
    ) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab, that also
        intersects the Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """
    @typing.overload
    def plus(
        self,
        val: typing.SupportsFloat,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of plus(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """
    @typing.overload
    def plus(
        self,
        mf: MultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function adds the values of the cells in mf to the corresponding
        cells of this MultiFab.  mf is required to have the same BoxArray or
        "valid region" as this MultiFab.  The addition is done only to num_comp
        components, starting with component number strt_comp.  The parameter
        nghost specifies the number of boundary cells that will be modified.
        If nghost == 0, only the valid region of each FArrayBox will be
        modified.
        """
    def saxpy(
        self,
        a: typing.SupportsFloat,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        self += a * src
        """
    @typing.overload
    def subtract(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def subtract(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def sum(self, comp: typing.SupportsInt = 0, local: bool = False) -> float:
        """
        Returns the sum of component 'comp' over the (i)MultiFab -- no ghost cells are included.
        """
    @typing.overload
    def sum(
        self, region: Box, comp: typing.SupportsInt = 0, local: bool = False
    ) -> float:
        """
        Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included.
        """
    @typing.overload
    def sum_unique(
        self,
        comp: typing.SupportsInt = 0,
        local: bool = False,
        period: Periodicity = ...,
    ) -> float:
        """
        Same as sum with local=false, but for non-cell-centered data, thisskips non-unique points that are owned by multiple boxes.
        """
    @typing.overload
    def sum_unique(
        self, region: Box, comp: typing.SupportsInt = 0, local: bool = False
    ) -> float:
        """
        Returns the unique sum of component `comp` in the given region. Non-unique points owned by multiple boxes in the MultiFab areonly added once. No ghost cells are included. This function does not takeperiodicity into account in the determination of uniqueness of points.
        """
    @typing.overload
    def swap(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """
    @typing.overload
    def swap(
        self,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into a MultiFab.

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of cupy.array
            A list of CuPy n-dimensional arrays, for each local block in the
            MultiFab.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into a MultiFab.

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of numpy.array
            A list of NumPy n-dimensional arrays, for each local block in the
            MultiFab.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a MultiFab,
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of xp.array
            A list of NumPy or CuPy n-dimensional arrays, for each local block in the
            MultiFab.

        """
    def weighted_sync(self, arg0: MultiFab, arg1: Periodicity) -> None: ...
    def xpay(
        self,
        a: typing.SupportsFloat,
        src: MultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        self = src + a * self
        """
    @property
    def n_comp(self) -> int: ...
    @property
    def n_grow_vect(self) -> IntVect2D: ...
    @property
    def shape(self, include_ghosts=False):
        """
        Returns the shape of the global array

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX
            include_ghosts : bool, default=False
                Whether or not ghost cells are included

        """
    @property
    def shape_with_ghosts(self):
        """
        Returns the shape of the global array including ghost cells

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX

        """

class PODVector_int_arena:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_arena) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_int_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_int_pinned:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_pinned) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_int_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_int_std:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_std) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_int_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_arena:
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_arena) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def assign(self, value: typing.SupportsFloat) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsFloat) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsFloat,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_real_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_pinned:
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_pinned) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def assign(self, value: typing.SupportsFloat) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsFloat) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsFloat,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_real_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_std:
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_std) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def assign(self, value: typing.SupportsFloat) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsFloat) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsFloat,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_real_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_uint64_arena:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_uint64_arena) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_uint64_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_uint64_pinned:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_uint64_pinned) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_uint64_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_uint64_std:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_uint64_std) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def assign(self, value: typing.SupportsInt) -> None:
        """
        assign the same value to every element
        """
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: typing.SupportsInt) -> None: ...
    def reserve(
        self,
        capacity: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    @typing.overload
    def resize(
        self,
        new_size: typing.SupportsInt,
        value: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a CuPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        cupy.array
            A 1D cupy array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> PODVector_uint64_pinned: ...
    def to_numpy(self, copy=False):
        """

        Provide a NumPy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D NumPy array.

        """
    def to_xp(self, copy=False):
        """

        Provide a NumPy or CuPy view into a PODVector (e.g., RealVector, IntVector),
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        xp.array
            A 1D NumPy or CuPy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ParConstIterBase_16_4_0_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_arena: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_arena: ...
    def soa(self) -> StructOfArrays_0_0_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_16_4_0_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_default: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_default: ...
    def soa(self) -> StructOfArrays_0_0_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_16_4_0_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_pinned: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_pinned: ...
    def soa(self) -> StructOfArrays_0_0_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_2_1_3_1_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_arena: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_arena: ...
    def soa(self) -> StructOfArrays_3_1_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_2_1_3_1_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_default: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_default: ...
    def soa(self) -> StructOfArrays_3_1_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_2_1_3_1_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_pinned: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_pinned: ...
    def soa(self) -> StructOfArrays_3_1_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_2_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_arena: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_2_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_default: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_2_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_pinned: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_6_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_arena: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_6_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_default: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_6_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_pinned: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_7_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_arena: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_7_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_default: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_7_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_pinned: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_8_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_arena: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_8_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_default: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIterBase_pureSoA_8_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_pinned: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParConstIter_16_4_0_0_arena(ParConstIterBase_16_4_0_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_16_4_0_0_default(ParConstIterBase_16_4_0_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_16_4_0_0_pinned(ParConstIterBase_16_4_0_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_2_1_3_1_arena(ParConstIterBase_2_1_3_1_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_2_1_3_1_default(ParConstIterBase_2_1_3_1_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_2_1_3_1_pinned(ParConstIterBase_2_1_3_1_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_2_0_arena(ParConstIterBase_pureSoA_2_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_2_0_default(ParConstIterBase_pureSoA_2_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_2_0_pinned(ParConstIterBase_pureSoA_2_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_6_0_arena(ParConstIterBase_pureSoA_6_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_6_0_default(ParConstIterBase_pureSoA_6_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_6_0_pinned(ParConstIterBase_pureSoA_6_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_7_0_arena(ParConstIterBase_pureSoA_7_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_7_0_default(ParConstIterBase_pureSoA_7_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_7_0_pinned(ParConstIterBase_pureSoA_7_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_0_arena(ParConstIterBase_pureSoA_8_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_0_default(ParConstIterBase_pureSoA_8_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_0_pinned(ParConstIterBase_pureSoA_8_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIterBase_16_4_0_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_arena: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_arena: ...
    def soa(self) -> StructOfArrays_0_0_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_16_4_0_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_default: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_default: ...
    def soa(self) -> StructOfArrays_0_0_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_16_4_0_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_16_4_pinned: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_16_4_0_0_pinned: ...
    def soa(self) -> StructOfArrays_0_0_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_2_1_3_1_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_arena: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_arena: ...
    def soa(self) -> StructOfArrays_3_1_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_2_1_3_1_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_default: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_default: ...
    def soa(self) -> StructOfArrays_3_1_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_2_1_3_1_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def aos(self) -> ArrayOfStructs_2_1_pinned: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_2_1_3_1_pinned: ...
    def soa(self) -> StructOfArrays_3_1_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_2_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_arena: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_2_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_default: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_2_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_2_0_pinned: ...
    def soa(self) -> StructOfArrays_2_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_6_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_arena: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_6_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_default: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_6_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_6_0_pinned: ...
    def soa(self) -> StructOfArrays_6_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_7_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_arena: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_7_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_default: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_7_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_7_0_pinned: ...
    def soa(self) -> StructOfArrays_7_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_8_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_arena: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_arena: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_8_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_default: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_default: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIterBase_pureSoA_8_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def _incr(self) -> None: ...
    def finalize(self) -> None: ...
    def geom(self, level: typing.SupportsInt) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_pureSoA_8_0_pinned: ...
    def soa(self) -> StructOfArrays_8_0_idcpu_pinned: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def level(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def pair_index(self) -> tuple[int, int]: ...
    @property
    def size(self) -> int:
        """
        the number of particles on this tile
        """

class ParIter_16_4_0_0_arena(ParIterBase_16_4_0_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_16_4_0_0_default(ParIterBase_16_4_0_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_16_4_0_0_pinned(ParIterBase_16_4_0_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_16_4_0_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_2_1_3_1_arena(ParIterBase_2_1_3_1_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_2_1_3_1_default(ParIterBase_2_1_3_1_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_2_1_3_1_pinned(ParIterBase_2_1_3_1_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_2_1_3_1_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_2_0_arena(ParIterBase_pureSoA_2_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_2_0_default(ParIterBase_pureSoA_2_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_2_0_pinned(ParIterBase_pureSoA_2_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_2_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_6_0_arena(ParIterBase_pureSoA_6_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_6_0_default(ParIterBase_pureSoA_6_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_6_0_pinned(ParIterBase_pureSoA_6_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_6_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_7_0_arena(ParIterBase_pureSoA_7_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_7_0_default(ParIterBase_pureSoA_7_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_7_0_pinned(ParIterBase_pureSoA_7_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_7_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_0_arena(ParIterBase_pureSoA_8_0_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_arena,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_0_default(ParIterBase_pureSoA_8_0_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_default,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_0_pinned(ParIterBase_pureSoA_8_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __getitem__(self, name):
        """
        Access (read/write) particle vectors.
        """
    def __init__(
        self,
        particle_container: ParticleContainer_pureSoA_8_0_pinned,
        level: typing.SupportsInt,
    ) -> None: ...
    def __iter__(self): ...
    def __next__(self):
        """
        This is a helper function for the C++ equivalent of void operator++()

            In Python, iterators always are called with __next__, even for the
            first access. This means we need to handle the first iterator element
            explicitly, otherwise we will jump directly to the 2nd element. We do
            this the same way as pybind11 does this, via a little state:
              https://github.com/AMReX-Codes/pyamrex/pull/50
              https://github.com/AMReX-Codes/pyamrex/pull/262
              https://github.com/pybind/pybind11/blob/v2.10.0/include/pybind11/pybind11.h#L2269-L2282

            Important: we must NOT copy the AMReX iterator (unnecessary and expensive).

            self: the current iterator
            returns: the updated iterator

        """
    def __repr__(self) -> str: ...

class ParmParse:
    @staticmethod
    def addfile(arg0: str) -> None: ...
    def __init__(self, prefix: str = "") -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def add(self, arg0: str, arg1: bool) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: str) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: IntVect2D) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: Box) -> None: ...
    @typing.overload
    def addarr(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...
    @typing.overload
    def addarr(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...
    @typing.overload
    def addarr(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...
    @typing.overload
    def addarr(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @typing.overload
    def addarr(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: collections.abc.Sequence[str]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: collections.abc.Sequence[IntVect2D]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: collections.abc.Sequence[Box]) -> None: ...
    def get_bool(self, name: str, ival: typing.SupportsInt = 0) -> bool:
        """
        parses input values
        """
    def get_int(self, name: str, ival: typing.SupportsInt = 0) -> int:
        """
        parses input values
        """
    def get_real(self, name: str, ival: typing.SupportsInt = 0) -> float:
        """
        parses input values
        """
    def pretty_print_table(self) -> None:
        """
        Write the table in a pretty way to the ostream. If there are duplicates, only the last one is printed.
        """
    def query_int(self, name: str, ival: typing.SupportsInt = 0) -> tuple[bool, int]:
        """
        queries input values
        """
    def remove(self, arg0: str) -> int: ...
    def to_dict(self) -> dict:
        """
        Convert to a nested Python dictionary.

        .. code-block:: python

            # Example: dump all ParmParse entries to YAML or TOML
            import toml
            import yaml

            pp = amr.ParmParse("").to_dict()
            yaml_string = yaml.dump(d)
            toml_string = toml.dumps(d)
        """

class ParticleContainer_16_4_0_0_arena:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 0
    num_struct_int: typing.ClassVar[int] = 4
    num_struct_real: typing.ClassVar[int] = 16
    ConstIterator = ParConstIter_16_4_0_0_arena
    Iterator = ParIter_16_4_0_0_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_16_4_0_0_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_16_4_0_0_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_16_4_0_0_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_16_4_0_0_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_16_4_0_0_default:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 0
    num_struct_int: typing.ClassVar[int] = 4
    num_struct_real: typing.ClassVar[int] = 16
    ConstIterator = ParConstIter_16_4_0_0_default
    Iterator = ParIter_16_4_0_0_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_16_4_0_0_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_16_4_0_0_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_16_4_0_0_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_16_4_0_0_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_16_4_0_0_pinned:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 0
    num_struct_int: typing.ClassVar[int] = 4
    num_struct_real: typing.ClassVar[int] = 16
    ConstIterator = ParConstIter_16_4_0_0_pinned
    Iterator = ParIter_16_4_0_0_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_16_4_0_0_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_16_4_0_0_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_16_4_0_0_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_16_4_0_0,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_16_4_0_0_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_2_1_3_1_arena:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 1
    num_array_real: typing.ClassVar[int] = 3
    num_struct_int: typing.ClassVar[int] = 1
    num_struct_real: typing.ClassVar[int] = 2
    ConstIterator = ParConstIter_2_1_3_1_arena
    Iterator = ParIter_2_1_3_1_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_2_1_3_1_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_2_1_3_1_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_2_1_3_1_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_2_1_3_1_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_2_1_3_1_default:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 1
    num_array_real: typing.ClassVar[int] = 3
    num_struct_int: typing.ClassVar[int] = 1
    num_struct_real: typing.ClassVar[int] = 2
    ConstIterator = ParConstIter_2_1_3_1_default
    Iterator = ParIter_2_1_3_1_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_2_1_3_1_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_2_1_3_1_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_2_1_3_1_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_2_1_3_1_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_2_1_3_1_pinned:
    is_soa_particle: typing.ClassVar[bool] = False
    num_array_int: typing.ClassVar[int] = 1
    num_array_real: typing.ClassVar[int] = 3
    num_struct_int: typing.ClassVar[int] = 1
    num_struct_real: typing.ClassVar[int] = 2
    ConstIterator = ParConstIter_2_1_3_1_pinned
    Iterator = ParIter_2_1_3_1_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_2_1_3_1_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_2_1_3_1_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_2_1_3_1_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_one_per_cell(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
        arg3: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def init_random_per_box(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_2_1_3_1,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_2_1_3_1_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_2_0_arena:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 2
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_2_0_arena
    Iterator = ParIter_pureSoA_2_0_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_2_0_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_2_0_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_2_0_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_2_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_2_0_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_2_0_default:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 2
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_2_0_default
    Iterator = ParIter_pureSoA_2_0_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_2_0_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_2_0_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_2_0_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_2_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_2_0_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_2_0_pinned:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 2
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_2_0_pinned
    Iterator = ParIter_pureSoA_2_0_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_2_0_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_2_0_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_2_0_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_2_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_2_0_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_6_0_arena:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 6
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_6_0_arena
    Iterator = ParIter_pureSoA_6_0_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_6_0_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_6_0_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_6_0_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_6_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_6_0_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_6_0_default:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 6
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_6_0_default
    Iterator = ParIter_pureSoA_6_0_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_6_0_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_6_0_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_6_0_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_6_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_6_0_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_6_0_pinned:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 6
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_6_0_pinned
    Iterator = ParIter_pureSoA_6_0_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_6_0_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_6_0_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_6_0_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_6_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_6_0_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_7_0_arena:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 7
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_7_0_arena
    Iterator = ParIter_pureSoA_7_0_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_7_0_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_7_0_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_7_0_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_7_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_7_0_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_7_0_default:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 7
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_7_0_default
    Iterator = ParIter_pureSoA_7_0_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_7_0_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_7_0_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_7_0_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_7_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_7_0_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_7_0_pinned:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 7
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_7_0_pinned
    Iterator = ParIter_pureSoA_7_0_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_7_0_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_7_0_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_7_0_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_7_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_7_0_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_8_0_arena:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 8
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_8_0_arena
    Iterator = ParIter_pureSoA_8_0_arena
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_8_0_arena, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_8_0_arena,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_8_0_arena]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_8_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_8_0_arena: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_8_0_default:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 8
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_8_0_default
    Iterator = ParIter_pureSoA_8_0_default
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_8_0_default, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_8_0_default,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_8_0_default]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_8_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_8_0_default: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleContainer_pureSoA_8_0_pinned:
    is_soa_particle: typing.ClassVar[bool] = True
    num_array_int: typing.ClassVar[int] = 0
    num_array_real: typing.ClassVar[int] = 8
    num_struct_int: typing.ClassVar[int] = 0
    num_struct_real: typing.ClassVar[int] = 0
    ConstIterator = ParConstIter_pureSoA_8_0_pinned
    Iterator = ParIter_pureSoA_8_0_pinned
    @typing.overload
    def Define(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def Define(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    def OK(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
    ) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_int,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: Vector_Geometry,
        arg1: Vector_DistributionMapping,
        arg2: Vector_BoxArray,
        arg3: Vector_IntVect,
    ) -> None: ...
    @typing.overload
    def add_int_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    @typing.overload
    def add_int_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Int
        """
    def add_particles(
        self, other: ParticleContainer_pureSoA_8_0_pinned, local: bool = False
    ) -> None: ...
    def add_particles_at_level(
        self,
        particles: ParticleTile_pureSoA_8_0_pinned,
        level: typing.SupportsInt,
        ngrow: typing.SupportsInt = 0,
    ) -> None: ...
    @typing.overload
    def add_real_comp(self, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    @typing.overload
    def add_real_comp(self, name: str, communicate: typing.SupportsInt = 1) -> None:
        """
        add a new runtime component with type Real
        """
    def clear_particles(self) -> None: ...
    def const_iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def get_int_comp_index(self, arg0: str) -> int:
        """
        Get the Integer SoA index of a component
        """
    def get_particles(
        self, level: typing.SupportsInt
    ) -> dict[tuple[int, int], ParticleTile_pureSoA_8_0_pinned]: ...
    def get_real_comp_index(self, arg0: str) -> int:
        """
        Get the ParticleReal SoA index of a component
        """
    def has_int_comp(self, arg0: str) -> bool:
        """
        Check if a container has an Integer component
        """
    def has_real_comp(self, arg0: str) -> bool:
        """
        Check if a container has an ParticleReal component
        """
    def increment(self, arg0: MultiFab, arg1: typing.SupportsInt) -> None: ...
    def init_random(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: ParticleInitType_pureSoA_8_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def iterator(self, *args, level=None):
        """
        Create an iterator over all particle tiles

            Parameters
            ----------
            self : amrex.ParticleContainer_*
                A ParticleContainer class in pyAMReX
            args : deprecated positional argument
            level : int | str, optional
                The MR level. Allowed values are [0:self.finest_level+1) and "all".
                If there is more than one MR level, the argument is required.

            Returns
            -------
            Iterator over all particle tiles at the specified level.

            Examples
            --------
            >>> pc.iterator(level="all")
            >>> pc.iterator(level=0)  # only particles on the the coarsest MR level

        """
    def make_alike(self) -> ParticleContainer_pureSoA_8_0_pinned: ...
    def num_local_tiles_at_level(self, level: typing.SupportsInt) -> int: ...
    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """
    def number_of_particles_at_level(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> int: ...
    def number_of_particles_in_grid(
        self,
        level: typing.SupportsInt,
        only_valid: bool = True,
        only_local: bool = False,
    ) -> Vector_Long: ...
    def print_capacity(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    def redistribute(
        self,
        lev_min: typing.SupportsInt = 0,
        lev_max: typing.SupportsInt = -1,
        nGrow: typing.SupportsInt = 0,
        local: typing.SupportsInt = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def remove_particles_at_level(self, arg0: typing.SupportsInt) -> None: ...
    def remove_particles_not_at_finestLevel(self) -> None: ...
    def reserve_data(self) -> None: ...
    def resize_data(self) -> None: ...
    def restart(self, dir: str, file: str) -> None: ...
    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...
    def set_soa_compile_time_names(
        self, arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[str]
    ) -> None: ...
    def shrink_t_fit(self) -> None: ...
    def sort_particles_by_bin(self, arg0: IntVect2D) -> None: ...
    def sort_particles_by_cell(self) -> None: ...
    def to_df(self, local=True, comm=None, root_rank=0):
        """

        Copy all particles into a pandas.DataFrame

        Parameters
        ----------
        self : amrex.ParticleContainer_*
            A ParticleContainer class in pyAMReX
        local : bool
            MPI rank-local particles only
        comm : MPI Communicator
            if local is False, this defaults to mpi4py.MPI.COMM_WORLD
        root_rank : MPI root rank to gather to
            if local is False, this defaults to 0

        Returns
        -------
        A concatenated pandas.DataFrame with particles from all levels.

        Returns None if no particles were found.
        If local=False, then all ranks but the root_rank will return None.

        """
    def total_number_of_particles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """
    def write_plotfile(self, dir: str, name: str) -> None: ...
    @property
    def byte_spread(self) -> typing.Annotated[list[int], "FixedSize(3)"]: ...
    @property
    def finest_level(self) -> int: ...
    @property
    def int_soa_names(self) -> list[str]:
        """
        Get the names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        The number of compile-time and runtime int components in SoA
        """
    @property
    def num_position_components(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        The number of compile-time and runtime Real components in SoA
        """
    @property
    def num_runtime_int_comps(self) -> int:
        """
        The number of runtime Int components in SoA
        """
    @property
    def num_runtime_real_comps(self) -> int:
        """
        The number of runtime Real components in SoA
        """
    @property
    def real_soa_names(self) -> list[str]:
        """
        Get the names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Return the number of valid particles on all MPI ranks
        """

class ParticleInitType_16_4_0_0:
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(0)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def int_struct_data(self) -> typing.Annotated[list[int], "FixedSize(4)"]: ...
    @int_struct_data.setter
    def int_struct_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(0)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def real_struct_data(self) -> typing.Annotated[list[float], "FixedSize(16)"]: ...
    @real_struct_data.setter
    def real_struct_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(16)"
        ],
    ) -> None: ...

class ParticleInitType_2_1_3_1:
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(1)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @property
    def int_struct_data(self) -> typing.Annotated[list[int], "FixedSize(1)"]: ...
    @int_struct_data.setter
    def int_struct_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(3)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    @property
    def real_struct_data(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    @real_struct_data.setter
    def real_struct_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...

class ParticleInitType_pureSoA_2_0:
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(0)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...

class ParticleInitType_pureSoA_6_0:
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(0)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(6)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
    ) -> None: ...

class ParticleInitType_pureSoA_7_0:
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(0)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(7)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(7)"
        ],
    ) -> None: ...

class ParticleInitType_pureSoA_8_0:
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(self) -> None: ...
    @property
    def int_array_data(self) -> typing.Annotated[list[int], "FixedSize(0)"]: ...
    @int_array_data.setter
    def int_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @property
    def real_array_data(self) -> typing.Annotated[list[float], "FixedSize(8)"]: ...
    @real_array_data.setter
    def real_array_data(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(8)"
        ],
    ) -> None: ...

class ParticleTileData_16_4_0_0:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def set_super_particle(
        self, arg0: Particle_16_4, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_2_1_3_1:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_5_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_5_2) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_5_2: ...
    def set_super_particle(
        self, arg0: Particle_5_2, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_pureSoA_2_0:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_0) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_2_0: ...
    def set_super_particle(
        self, arg0: Particle_2_0, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_pureSoA_6_0:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_6_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_6_0) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_6_0: ...
    def set_super_particle(
        self, arg0: Particle_6_0, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_pureSoA_7_0:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_7_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_7_0) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_7_0: ...
    def set_super_particle(
        self, arg0: Particle_7_0, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_pureSoA_8_0:
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_8_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_8_0) -> None: ...
    def get_super_particle(self, arg0: typing.SupportsInt) -> Particle_8_0: ...
    def set_super_particle(
        self, arg0: Particle_8_0, arg1: typing.SupportsInt
    ) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTile_16_4_0_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 0
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_16_4_arena: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_0_0_arena: ...
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_16_4_0_0_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_16_4_0_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 0
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_16_4_default: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_0_0_default: ...
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_16_4_0_0_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_16_4_0_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 0
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_16_4: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_16_4) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_16_4_pinned: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_0_0_pinned: ...
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_16_4) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_16_4_0_0_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_2_1_3_1_arena:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 3
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_5_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_5_2) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_2_1_arena: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...
    def get_struct_of_arrays(self) -> StructOfArrays_3_1_arena: ...
    @typing.overload
    def push_back(self, arg0: Particle_2_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_2_1_3_1_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_2_1_3_1_default:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 3
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_5_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_5_2) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_2_1_default: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...
    def get_struct_of_arrays(self) -> StructOfArrays_3_1_default: ...
    @typing.overload
    def push_back(self, arg0: Particle_2_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_2_1_3_1_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_2_1_3_1_pinned:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 3
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_5_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_5_2) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_array_of_structs(self) -> ArrayOfStructs_2_1_pinned: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...
    def get_struct_of_arrays(self) -> StructOfArrays_3_1_pinned: ...
    @typing.overload
    def push_back(self, arg0: Particle_2_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_2_1_3_1_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_2_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 2
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_2_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_2_0_idcpu_arena: ...
    def push_back(self, arg0: Particle_2_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_2_0_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_2_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 2
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_2_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_2_0_idcpu_default: ...
    def push_back(self, arg0: Particle_2_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_2_0_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_2_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 2
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_2_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_2_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_2_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_2_0_idcpu_pinned: ...
    def push_back(self, arg0: Particle_2_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_2_0_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_6_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 6
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_6_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_6_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_6_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_6_0_idcpu_arena: ...
    def push_back(self, arg0: Particle_6_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_6_0_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_6_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 6
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_6_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_6_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_6_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_6_0_idcpu_default: ...
    def push_back(self, arg0: Particle_6_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_6_0_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_6_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 6
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_6_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_6_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_6_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_6_0_idcpu_pinned: ...
    def push_back(self, arg0: Particle_6_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_6_0_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_7_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 7
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_7_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_7_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_7_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_7_0_idcpu_arena: ...
    def push_back(self, arg0: Particle_7_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(7)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_7_0_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_7_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 7
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_7_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_7_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_7_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_7_0_idcpu_default: ...
    def push_back(self, arg0: Particle_7_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(7)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_7_0_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_7_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 7
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_7_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_7_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_7_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_7_0_idcpu_pinned: ...
    def push_back(self, arg0: Particle_7_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(7)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_7_0_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_8_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 8
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_8_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_8_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_8_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_8_0_idcpu_arena: ...
    def push_back(self, arg0: Particle_8_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(8)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_8_0_arena) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_8_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 8
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_8_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_8_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_8_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_8_0_idcpu_default: ...
    def push_back(self, arg0: Particle_8_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(8)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_8_0_default) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParticleTile_pureSoA_8_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 8
    def __getitem__(self, arg0: typing.SupportsInt) -> Particle_8_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Particle_8_0) -> None: ...
    def capacity(self) -> int: ...
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
        arg4: Arena,
    ) -> None: ...
    def get_num_neighbors(self) -> int: ...
    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_8_0: ...
    def get_struct_of_arrays(self) -> StructOfArrays_8_0_idcpu_pinned: ...
    def push_back(self, arg0: Particle_8_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_int(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsInt,
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(8)"
        ],
    ) -> None: ...
    @typing.overload
    def push_back_real(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    def resize(
        self,
        count: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def swap(self, arg0: ParticleTile_pureSoA_8_0_pinned) -> None: ...
    @property
    def empty(self) -> bool: ...
    @property
    def num_int_comps(self) -> int: ...
    @property
    def num_neighbor_particles(self) -> int: ...
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int: ...
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_runtime_int_comps(self) -> int: ...
    @property
    def num_runtime_real_comps(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def size(self) -> int: ...

class Particle_16_4:
    NInt: typing.ClassVar[int] = 4
    NReal: typing.ClassVar[int] = 16
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def get_idata(self) -> typing.Annotated[list[int], "FixedSize(4)"]: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(16)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(4)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(16)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_2_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 2
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_2_1:
    NInt: typing.ClassVar[int] = 1
    NReal: typing.ClassVar[int] = 2
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def get_idata(self) -> typing.Annotated[list[int], "FixedSize(1)"]: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(1)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_5_2:
    NInt: typing.ClassVar[int] = 2
    NReal: typing.ClassVar[int] = 5
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def get_idata(self) -> typing.Annotated[list[int], "FixedSize(2)"]: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(5)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(5)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_6_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 6
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(6)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_7_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 7
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(7)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(7)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Particle_8_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 8
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, *args
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, **kwargs
    ) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: typing.SupportsInt) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def get_rdata(self) -> typing.Annotated[list[float], "FixedSize(8)"]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def set_idata(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None: ...
    @typing.overload
    def set_idata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(0)"
        ],
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def set_rdata(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(8)"
        ],
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None: ...

class Periodicity:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def non_periodic() -> Periodicity:
        """
        Return the Periodicity object that is not periodic in any direction
        """
    def __eq__(self, arg0: Periodicity) -> bool: ...
    def __getitem__(self, dir: typing.SupportsInt) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: IntVect2D) -> None: ...
    def __repr__(self) -> str: ...
    def is_periodic(self, dir: typing.SupportsInt) -> bool: ...
    @property
    def domain(self) -> Box:
        """
        Cell-centered domain Box "infinitely" long in non-periodic directions.
        """
    @property
    def is_all_periodic(self) -> bool: ...
    @property
    def is_any_periodic(self) -> bool: ...
    @property
    def shift_IntVect(self, arg1: IntVect2D) -> list[IntVect2D]: ...

class PlotFileData:
    def DistributionMap(self, arg0: typing.SupportsInt) -> DistributionMapping: ...
    def __init__(self, arg0: str) -> None: ...
    def boxArray(self, arg0: typing.SupportsInt) -> BoxArray: ...
    def cellSize(
        self, arg0: typing.SupportsInt
    ) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def coordSys(self) -> int: ...
    def finestLevel(self) -> int: ...
    @typing.overload
    def get(self, arg0: typing.SupportsInt) -> MultiFab: ...
    @typing.overload
    def get(self, arg0: typing.SupportsInt, arg1: str) -> MultiFab: ...
    def levelStep(self, arg0: typing.SupportsInt) -> int: ...
    def nComp(self) -> int: ...
    def nGrowVect(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    def probDomain(self, arg0: typing.SupportsInt) -> Box: ...
    def probHi(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def probLo(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def probSize(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    def refRatio(self, arg0: typing.SupportsInt) -> int: ...
    def spaceDim(self) -> int: ...
    @typing.overload
    def syncDistributionMap(self, arg0: PlotFileData) -> None: ...
    @typing.overload
    def syncDistributionMap(
        self, arg0: typing.SupportsInt, arg1: PlotFileData
    ) -> None: ...
    def time(self) -> float: ...
    def varNames(self) -> Vector_string: ...

class RealBox:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        x_lo: typing.SupportsFloat,
        y_lo: typing.SupportsFloat,
        x_hi: typing.SupportsFloat,
        y_hi: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        a_lo: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
        a_hi: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        bx: Box,
        dx: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
        base: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str(self) -> str: ...
    @typing.overload
    def contains(self, rb: XDim3, eps: typing.SupportsFloat = 0.0) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def contains(self, rb: RealVect, eps: typing.SupportsFloat = 0.0) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def contains(self, rb: RealBox, eps: typing.SupportsFloat = 0.0) -> bool:
        """
        Determine if RealBox contains another RealBox, within tolerance ``eps``
        """
    @typing.overload
    def contains(
        self,
        rb: collections.abc.Sequence[typing.SupportsFloat],
        eps: typing.SupportsFloat = 0.0,
    ) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def hi(self, arg0: typing.SupportsInt) -> float:
        """
        Get ith component of ``xhi``
        """
    @typing.overload
    def hi(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Get all components of ``xhi``
        """
    def intersects(self, arg0: RealBox) -> bool:
        """
        determine if box intersects with a box
        """
    def length(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def lo(self, arg0: typing.SupportsInt) -> float:
        """
        Get ith component of ``xlo``
        """
    @typing.overload
    def lo(self) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Get all components of ``xlo``
        """
    def ok(self) -> bool:
        """
        Determine if RealBox satisfies ``xlo[i]<xhi[i]`` for ``i=0,1,...,AMREX_SPACEDIM``.
        """
    @typing.overload
    def setHi(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Get all components of ``xlo``
        """
    @typing.overload
    def setHi(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        """
        Get ith component of ``xhi``
        """
    @typing.overload
    def setLo(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Get ith component of ``xlo``
        """
    @typing.overload
    def setLo(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        """
        Get all components of ``xlo``
        """
    def volume(self) -> float: ...
    @property
    def xhi(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...
    @property
    def xlo(self) -> typing.Annotated[list[float], "FixedSize(2)"]: ...

class RealVect:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def unit_vector() -> RealVect: ...
    @staticmethod
    def zero_vector() -> RealVect: ...
    def BASISREALV(self: typing.SupportsInt) -> RealVect:
        """
        return basis vector in given coordinate direction
        """
    @typing.overload
    def __add__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __add__(self, arg0: RealVect) -> RealVect: ...
    def __eq__(self, arg0: RealVect) -> bool: ...
    def __ge__(self, arg0: RealVect) -> bool: ...
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    def __gt__(self, arg0: RealVect) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __iadd__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __imul__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __imul__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: IntVect2D) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsFloat) -> None: ...
    @typing.overload
    def __isub__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __isub__(self, arg0: RealVect) -> RealVect: ...
    def __itruediv__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __le__(self, arg0: RealVect) -> bool: ...
    def __lt__(self, arg0: RealVect) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __ne__(self, arg0: RealVect) -> bool: ...
    def __neg__(self) -> RealVect: ...
    def __pos__(self) -> RealVect: ...
    def __radd__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __rsub__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __rtruediv__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> float: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __sub__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsFloat) -> RealVect: ...
    @typing.overload
    def __truediv__(self, arg0: RealVect) -> RealVect: ...
    def ceil(self) -> IntVect2D:
        """
        Return an ``IntVect`` whose components are the std::ceil of the vector components
        """
    def dotProduct(self, arg0: RealVect) -> float:
        """
        Return dot product of this vector with another
        """
    def floor(self) -> IntVect2D:
        """
        Return an ``IntVect`` whose components are the std::floor of the vector components
        """
    def max(self, arg0: RealVect) -> RealVect:
        """
        Replace vector with the component-wise maxima of this vector and another
        """
    def maxDir(self, arg0: bool) -> int:
        """
        direction or index of maximum value of this vector
        """
    def min(self, arg0: RealVect) -> RealVect:
        """
        Replace vector with the component-wise minima of this vector and another
        """
    def minDir(self, arg0: bool) -> int:
        """
        direction or index of minimum value of this vector
        """
    def round(self) -> IntVect2D:
        """
        Return an ``IntVect`` whose components are the std::round of the vector components
        """
    def scale(self, arg0: typing.SupportsFloat) -> RealVect:
        """
        Multiplify each component of this vector by a scalar
        """
    @property
    def product(self) -> float:
        """
        Product of entries of this vector
        """
    @property
    def radSquared(self) -> float:
        """
        Length squared of this vector
        """
    @property
    def sum(self) -> float:
        """
        Sum of the components of this vector
        """
    @property
    def vectorLength(self) -> float:
        """
        Length or 2-Norm of this vector
        """

class SmallMatrix_1x6_F_SI1_double:
    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_double: ...
    def __add__(
        self, arg0: SmallMatrix_1x6_F_SI1_double
    ) -> SmallMatrix_1x6_F_SI1_double: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_1x6_F_SI1_double) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> None: ...
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_double: ...
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_double: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_double: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_1x6_F_SI1_double
    ) -> SmallMatrix_1x6_F_SI1_double: ...
    def dot(self, arg0: SmallMatrix_1x6_F_SI1_double) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_double: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_double: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_1x6_F_SI1_float:
    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_float: ...
    def __add__(
        self, arg0: SmallMatrix_1x6_F_SI1_float
    ) -> SmallMatrix_1x6_F_SI1_float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_1x6_F_SI1_float) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> None: ...
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_float: ...
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_float: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_float: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_1x6_F_SI1_float
    ) -> SmallMatrix_1x6_F_SI1_float: ...
    def dot(self, arg0: SmallMatrix_1x6_F_SI1_float) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_1x6_F_SI1_float: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_float: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_1x6_F_SI1_longdouble:
    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def __add__(
        self, arg0: SmallMatrix_1x6_F_SI1_longdouble
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_1x6_F_SI1_longdouble) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.longdouble]
    ) -> None: ...
    def __mul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def __repr__(self) -> str: ...
    def __rmul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_1x6_F_SI1_longdouble
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def dot(self, arg0: SmallMatrix_1x6_F_SI1_longdouble) -> float: ...
    def prod(self) -> float: ...
    def set_val(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x1_F_SI1_double:
    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_double: ...
    def __add__(
        self, arg0: SmallMatrix_6x1_F_SI1_double
    ) -> SmallMatrix_6x1_F_SI1_double: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x1_F_SI1_double) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> None: ...
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_double: ...
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_double: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_double: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x1_F_SI1_double
    ) -> SmallMatrix_6x1_F_SI1_double: ...
    def dot(self, arg0: SmallMatrix_6x1_F_SI1_double) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_double: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_double: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x1_F_SI1_float:
    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_float: ...
    def __add__(
        self, arg0: SmallMatrix_6x1_F_SI1_float
    ) -> SmallMatrix_6x1_F_SI1_float: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x1_F_SI1_float) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> None: ...
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_float: ...
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_float: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_float: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x1_F_SI1_float
    ) -> SmallMatrix_6x1_F_SI1_float: ...
    def dot(self, arg0: SmallMatrix_6x1_F_SI1_float) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x1_F_SI1_float: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_float: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x1_F_SI1_longdouble:
    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def __add__(
        self, arg0: SmallMatrix_6x1_F_SI1_longdouble
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    @typing.overload
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x1_F_SI1_longdouble) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.longdouble]
    ) -> None: ...
    def __mul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def __repr__(self) -> str: ...
    def __rmul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    @typing.overload
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x1_F_SI1_longdouble
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def dot(self, arg0: SmallMatrix_6x1_F_SI1_longdouble) -> float: ...
    def prod(self) -> float: ...
    def set_val(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x6_F_SI1_double:
    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_double: ...
    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_double: ...
    def __add__(
        self, arg0: SmallMatrix_6x6_F_SI1_double
    ) -> SmallMatrix_6x6_F_SI1_double: ...
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x6_F_SI1_double) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> None: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_double: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x6_F_SI1_double
    ) -> SmallMatrix_6x6_F_SI1_double: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x1_F_SI1_double
    ) -> SmallMatrix_6x1_F_SI1_double: ...
    def __neg__(self) -> SmallMatrix_6x6_F_SI1_double: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_double: ...
    @typing.overload
    def __rmul__(
        self, arg0: SmallMatrix_1x6_F_SI1_double
    ) -> SmallMatrix_1x6_F_SI1_double: ...
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x6_F_SI1_double
    ) -> SmallMatrix_6x6_F_SI1_double: ...
    def dot(self, arg0: SmallMatrix_6x6_F_SI1_double) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_double: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    def trace(self) -> float: ...
    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_double: ...
    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_double: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x6_F_SI1_float:
    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_float: ...
    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_float: ...
    def __add__(
        self, arg0: SmallMatrix_6x6_F_SI1_float
    ) -> SmallMatrix_6x6_F_SI1_float: ...
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x6_F_SI1_float) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> None: ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_float: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x6_F_SI1_float
    ) -> SmallMatrix_6x6_F_SI1_float: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x1_F_SI1_float
    ) -> SmallMatrix_6x1_F_SI1_float: ...
    def __neg__(self) -> SmallMatrix_6x6_F_SI1_float: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __rmul__(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_float: ...
    @typing.overload
    def __rmul__(
        self, arg0: SmallMatrix_1x6_F_SI1_float
    ) -> SmallMatrix_1x6_F_SI1_float: ...
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x6_F_SI1_float
    ) -> SmallMatrix_6x6_F_SI1_float: ...
    def dot(self, arg0: SmallMatrix_6x6_F_SI1_float) -> float: ...
    def prod(self) -> float: ...
    def set_val(self, arg0: typing.SupportsFloat) -> SmallMatrix_6x6_F_SI1_float: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    def trace(self) -> float: ...
    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_float: ...
    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_float: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class SmallMatrix_6x6_F_SI1_longdouble:
    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_longdouble: ...
    def __add__(
        self, arg0: SmallMatrix_6x6_F_SI1_longdouble
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    def __getitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
    ) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: SmallMatrix_6x6_F_SI1_longdouble) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.longdouble]
    ) -> None: ...
    @typing.overload
    def __mul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x6_F_SI1_longdouble
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @typing.overload
    def __mul__(
        self, arg0: SmallMatrix_6x1_F_SI1_longdouble
    ) -> SmallMatrix_6x1_F_SI1_longdouble: ...
    def __neg__(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __rmul__(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @typing.overload
    def __rmul__(
        self, arg0: SmallMatrix_1x6_F_SI1_longdouble
    ) -> SmallMatrix_1x6_F_SI1_longdouble: ...
    def __setitem__(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"
        ],
        arg1: typing.SupportsFloat,
    ) -> None: ...
    def __sub__(
        self, arg0: SmallMatrix_6x6_F_SI1_longdouble
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    def dot(self, arg0: SmallMatrix_6x6_F_SI1_longdouble) -> float: ...
    def prod(self) -> float: ...
    def set_val(
        self, arg0: typing.SupportsFloat
    ) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    def sum(self) -> float: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        cupy.array
            A cupy 2-dimensional array.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into an SmallMatrix.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        np.array
            A NumPy 2-dimensional array.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a SmallMatrix, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.SmallMatrix_*
            A SmallMatrix class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        xp.array
            A NumPy or CuPy 2-dimensional array.

        """
    def trace(self) -> float: ...
    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    @property
    def column_size(self) -> int: ...
    @property
    def order(self) -> str: ...
    @property
    def row_size(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def starting_index(self) -> int: ...

class StructOfArrays_0_0_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(0)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_0_0_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(0)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_0_0_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_2_0_idcpu_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_arena:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(2)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_2_0_idcpu_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_std:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(2)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_2_0_idcpu_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_pinned:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(2)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_3_1_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(1)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(3)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_3_1_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(1)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(3)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_3_1_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(1)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(3)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_6_0_idcpu_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_arena:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(6)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_6_0_idcpu_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_std:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(6)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_6_0_idcpu_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_pinned:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(6)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_7_0_idcpu_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_arena:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(7)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_7_0_idcpu_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_std:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(7)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_7_0_idcpu_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_pinned:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(7)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_8_0_idcpu_arena:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_arena:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_arena], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_arena], "FixedSize(8)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_8_0_idcpu_default:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_std:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(self) -> typing.Annotated[list[PODVector_int_std], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_std], "FixedSize(8)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class StructOfArrays_8_0_idcpu_pinned:
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: collections.abc.Sequence[str],
        arg3: collections.abc.Sequence[str],
    ) -> None: ...
    def get_idcpu_data(self) -> PODVector_uint64_pinned:
        """
        Get access to a particle IdCPU component Array
        """
    @typing.overload
    def get_int_data(
        self,
    ) -> typing.Annotated[list[PODVector_int_pinned], "FixedSize(0)"]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def get_int_data(self, index: typing.SupportsInt) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def get_num_neighbors(self) -> int: ...
    @typing.overload
    def get_real_data(
        self,
    ) -> typing.Annotated[list[PODVector_real_pinned], "FixedSize(8)"]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def get_real_data(self, index: typing.SupportsInt) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def resize(
        self,
        new_size: typing.SupportsInt,
        strategy: GrowthStrategy = amrex.space2d.GrowthStrategy.Poisson,
    ) -> None: ...
    def set_num_neighbors(self, arg0: typing.SupportsInt) -> None: ...
    def to_cupy(self, copy=False):
        """

        Provide CuPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide NumPy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    def to_xp(self, copy=False):
        """

        Provide NumPy or CuPy views into a StructOfArrays, depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each dicts
            of 1D NumPy or CuPy arrays. The dictionary key order is the same as
            in the C++ component order.
            For pure SoA particle layouts, an additional component idcpu
            with global particle indices is populated.

        """
    @property
    def has_idcpu(self) -> bool:
        """
        In pure SoA particle layout, idcpu is an array in the SoA
        """
    @property
    def int_names(self) -> list[str]:
        """
        Names for the int SoA components
        """
    @property
    def num_int_comps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    @property
    def num_particles(self) -> int: ...
    @property
    def num_real_comps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    @property
    def num_real_particles(self) -> int: ...
    @property
    def num_total_particles(self) -> int: ...
    @property
    def real_names(self) -> list[str]:
        """
        Names for the Real SoA components
        """
    @property
    def size(self) -> int:
        """
        Get the number of particles
        """

class Vector_Box:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: Box) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_Box) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_Box:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> Box: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> Box: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Box) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Box) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[Box]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_Box) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Box) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Box) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Box) -> None: ...
    def append(self, x: Box) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: Box) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_Box) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: Box) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> Box:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> Box:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: Box) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class Vector_BoxArray:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: BoxArray) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_BoxArray) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_BoxArray:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> BoxArray: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_BoxArray) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_BoxArray) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[BoxArray]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_BoxArray) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: BoxArray) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_BoxArray) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: BoxArray) -> None: ...
    def append(self, x: BoxArray) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: BoxArray) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_BoxArray) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: BoxArray) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> BoxArray:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> BoxArray:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: BoxArray) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class Vector_DistributionMapping:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: DistributionMapping) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_DistributionMapping) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_DistributionMapping:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> DistributionMapping: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> DistributionMapping: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_DistributionMapping) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_DistributionMapping) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[DistributionMapping]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_DistributionMapping) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: DistributionMapping
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_DistributionMapping) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: DistributionMapping
    ) -> None: ...
    def append(self, x: DistributionMapping) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: DistributionMapping) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_DistributionMapping) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: DistributionMapping) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> DistributionMapping:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> DistributionMapping:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: DistributionMapping) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class Vector_Geometry:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_Geometry:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> Geometry: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> Geometry: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Geometry) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Geometry) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[Geometry]: ...
    def __len__(self) -> int: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Geometry) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Geometry) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: Geometry) -> None: ...
    def append(self, x: Geometry) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: Vector_Geometry) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: Geometry) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> Geometry:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> Geometry:
        """
        Remove and return the item at index ``i``
        """
    def size(self) -> int: ...

class Vector_IntVect:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: IntVect2D) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_IntVect) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_IntVect:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> IntVect2D: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_IntVect) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_IntVect) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[IntVect2D]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_IntVect) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: IntVect2D) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_IntVect) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: IntVect2D) -> None: ...
    def append(self, x: IntVect2D) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: IntVect2D) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_IntVect) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: IntVect2D) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> IntVect2D:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> IntVect2D:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: IntVect2D) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class Vector_Long:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: typing.SupportsInt) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_Long) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_Long:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Long) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Long) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_Long) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Long) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def append(self, x: typing.SupportsInt) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: typing.SupportsInt) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_Long) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: typing.SupportsInt) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: typing.SupportsInt) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_Real:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: typing.SupportsFloat) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_Real) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_Real:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Real) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Real) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[float]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_Real) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Real) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> None: ...
    def append(self, x: typing.SupportsFloat) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: typing.SupportsFloat) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_Real) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: typing.SupportsFloat) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> float:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> float:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: typing.SupportsFloat) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_int:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: typing.SupportsInt) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_int) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_int:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[int]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_int) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_int) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> None: ...
    def append(self, x: typing.SupportsInt) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: typing.SupportsInt) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_int) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: typing.SupportsInt) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: typing.SupportsInt) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_string:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: str) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector_string) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector_string:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> str: ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> str: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_string) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_string) -> None: ...
    def __iter__(self) -> collections.abc.Iterator[str]: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: Vector_string) -> bool: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: str) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_string) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: str) -> None: ...
    def append(self, x: str) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: str) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_string) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: str) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> str:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> str:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: str) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class VisMF:
    @staticmethod
    @typing.overload
    def Read(name: str) -> MultiFab:
        """
        Reads a MultiFab from the specified file
        """
    @staticmethod
    @typing.overload
    def Read(name: str, mf: MultiFab) -> None:
        """
        Reads a MultiFab from the specified file into the given MultiFab. The BoxArray on the disk must match the BoxArray * in mf
        """
    @staticmethod
    def Write(mf: FabArray_FArrayBox, name: str) -> int:
        """
        Writes a Multifab to the specified file
        """

class XDim3:
    def __init__(
        self,
        arg0: typing.SupportsFloat,
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsFloat,
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat) -> None: ...

class iMultiFab(FabArray_IArrayBox):
    @staticmethod
    def __iter__(imfab): ...
    @staticmethod
    def finalize() -> None: ...
    @staticmethod
    def initialize() -> None: ...
    def __getitem__(self, index, with_internal_ghosts=False):
        """
        Returns slice of the MultiFab using global indexing, as a numpy array.
            This uses numpy array indexing, with the indexing relative to the global array.
            The slice ranges can cross multiple blocks and the result will be gathered into a single
            array.

            In an MPI context, this is a global operation. An "allgather" is performed so that the full
            result is returned on all processors.

            Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

            The default range of the indices includes only the valid cells. The ":" index will include all of
            the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
            negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
            The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
            ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

            Parameters
            ----------
            index : the index using numpy style indexing
                Index of the slice to return.
            with_internal_ghosts : bool, optional
                Whether to include internal ghost cells. When true, data from ghost cells may be used that
                overlaps valid cells.

        """
    @typing.overload
    def __init__(self) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions
                    inherited from FabArray.
        """
    @typing.overload
    def __init__(self, a: Arena) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions.
                    If ``define`` is called later with a nullptr as MFInfo's arena, the
                    default Arena ``a`` will be used.  If the arena in MFInfo is not a
                    nullptr, the MFInfo's arena will be used.
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
        info: MFInfo,
        factory: FabFactory_IArrayBox,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
        info: MFInfo,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: typing.SupportsInt,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        info: MFInfo,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
        info: MFInfo,
        factory: FabFactory_IArrayBox,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    @typing.overload
    def __init__(
        self,
        bxs: BoxArray,
        dm: DistributionMapping,
        ncomp: typing.SupportsInt,
        ngrow: IntVect2D,
    ) -> None:
        """
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \\p ngrow, and
        the number of components is given by \\p ncomp. If \\p info is set to
        not allocating memory, then no FArrayBoxes are allocated at
        this time but can be defined later.

        Parameters
        ----------
        bxs :
          a valid region
        dm :
          a DistributionMapping
        ncomp :
          number of components
        ngrow :
          number of cells the region grows
        info :
          (i)MultiFab info, including allocation Arena
        factory :
          FArrayBoxFactory for embedded boundaries
        """
    def __repr__(self) -> str: ...
    def __setitem__(self, index, value):
        """
        Sets the slice of the MultiFab using global indexing.
            This uses numpy array indexing, with the indexing relative to the global array.
            The slice ranges can cross multiple blocks and the value will be distributed accordingly.
            Note that this will apply the value to both valid and ghost cells.

            In an MPI context, this is a local operation. On each processor, the blocks within the slice
            range will be set to the value.

            Note that the index is in fortran ordering and that 0 is the lower boundary of the whole domain.

            The default range of the indices includes only the valid cells. The ":" index will include all of
            the valid cels and no ghost cells. The ghost cells can be accessed using imaginary numbers, with
            negative imaginary numbers for the lower ghost cells, and positive for the upper ghost cells.
            The index "[-1j]" for example refers to the first lower ghost cell, and "[1j]" to the first upper
            ghost cell. To access all cells, ghosts and valid cells, use an empty tuple for the index, i.e. "[()]".

            Parameters
            ----------
            index : the index using numpy style indexing
                Index of the slice to return.
            value : scalar or array
                Input value to assign to the specified slice of the MultiFab

        """
    @typing.overload
    def add(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def add(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    def box_array(self: FabArrayBase) -> BoxArray: ...
    def copy(self):
        """

        Create a copy of this MultiFab, using the same Arena.

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX

        Returns
        -------
        amrex.MultiFab
            A copy of this MultiFab.

        """
    def copymf(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        dstcomp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Copy from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray. The copy is local
        """
    def divi(
        self,
        mf: iMultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function divides the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The division is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.  Note, nothing is done to protect
        against divide by zero.
        """
    @typing.overload
    def divide(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def divide(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    def dm(self: FabArrayBase) -> DistributionMapping: ...
    def imesh(self, idir, include_ghosts=False):
        """
        Returns the integer mesh along the specified direction with the appropriate centering.
            This is the location of the data points in grid cell units.

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX
            direction : integer
                Zero based direction number.
                In a typical Cartesian case, 0 would be 'x' direction.
            include_ghosts : bool, default=False
                Whether or not ghost cells are included in the mesh.

        """
    @typing.overload
    def max(
        self,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> int:
        """
        Returns the maximum value of the specified component of the (i)MultiFab.
        """
    @typing.overload
    def max(
        self,
        region: Box,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> int:
        """
        Returns the maximum value of the specified component of the (i)MultiFab over the region.
        """
    def maxIndex(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> IntVect2D: ...
    @typing.overload
    def min(
        self,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> int:
        """
        Returns the minimum value of the specified component of the (i)MultiFab.
        """
    @typing.overload
    def min(
        self,
        region: Box,
        comp: typing.SupportsInt = 0,
        nghost: typing.SupportsInt = 0,
        local: bool = False,
    ) -> int:
        """
        Returns the minimum value of the specified component of the (i)MultiFab over the region.
        """
    def minIndex(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> IntVect2D: ...
    def minus(
        self,
        mf: iMultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function subtracts the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The subtraction is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.
        """
    @typing.overload
    def mult(self, val: typing.SupportsInt, nghost: typing.SupportsInt = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val).
        The value of nghost specifies the number of cells in the
        boundary region that should be modified.
        """
    @typing.overload
    def mult(
        self,
        val: typing.SupportsInt,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Scales the value of each cell in the specified subregion of the
        MultiFab by the scalar val (a[i] <- a[i]*val). The subregion
        consists of the num_comp components starting at component comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """
    @typing.overload
    def mult(
        self,
        val: typing.SupportsInt,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of mult(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """
    @typing.overload
    def mult(
        self, val: typing.SupportsInt, region: Box, nghost: typing.SupportsInt = 0
    ) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """
    @typing.overload
    def multiply(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def multiply(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def negate(self, nghost: typing.SupportsInt = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab.  The value of nghost specifies the number of
        cells in the boundary region that should be modified.
        """
    @typing.overload
    def negate(
        self,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Negates the value of each cell in the specified subregion of
        the MultiFab.  The subregion consists of the num_comp
        components starting at component comp.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """
    @typing.overload
    def negate(self, region: Box, nghost: typing.SupportsInt = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab that also intersects the Box region.  The value
        of nghost specifies the number of cells in the boundary region
        that should be modified.
        """
    @typing.overload
    def negate(
        self,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of negate(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """
    @typing.overload
    def plus(self, val: typing.SupportsInt, nghost: typing.SupportsInt = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab.  The value
        of nghost specifies the number of cells in the boundary
        region that should be modified.
        """
    @typing.overload
    def plus(
        self,
        val: typing.SupportsInt,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Adds the scalar value \\p val to the value of each cell in the
        specified subregion of the MultiFab.

        The subregion consists of the \\p num_comp components starting at component \\p comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """
    @typing.overload
    def plus(
        self, val: typing.SupportsInt, region: Box, nghost: typing.SupportsInt = 0
    ) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab, that also
        intersects the Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """
    @typing.overload
    def plus(
        self,
        val: typing.SupportsInt,
        region: Box,
        comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        Identical to the previous version of plus(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """
    @typing.overload
    def plus(
        self,
        mf: iMultiFab,
        strt_comp: typing.SupportsInt,
        num_comp: typing.SupportsInt,
        nghost: typing.SupportsInt = 0,
    ) -> None:
        """
        This function adds the values of the cells in mf to the corresponding
        cells of this MultiFab.  mf is required to have the same BoxArray or
        "valid region" as this MultiFab.  The addition is done only to num_comp
        components, starting with component number strt_comp.  The parameter
        nghost specifies the number of boundary cells that will be modified.
        If nghost == 0, only the valid region of each FArrayBox will be
        modified.
        """
    @typing.overload
    def subtract(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def subtract(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """
    @typing.overload
    def sum(self, comp: typing.SupportsInt = 0, local: bool = False) -> int:
        """
        Returns the sum of component 'comp' over the (i)MultiFab -- no ghost cells are included.
        """
    @typing.overload
    def sum(
        self, region: Box, comp: typing.SupportsInt = 0, local: bool = False
    ) -> int:
        """
        Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included.
        """
    @typing.overload
    def swap(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: typing.SupportsInt,
    ) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """
    @typing.overload
    def swap(
        self,
        src: iMultiFab,
        srccomp: typing.SupportsInt,
        comp: typing.SupportsInt,
        numcomp: typing.SupportsInt,
        nghost: IntVect2D,
    ) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a CuPy view into a MultiFab.

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of cupy.array
            A list of CuPy n-dimensional arrays, for each local block in the
            MultiFab.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a NumPy view into a MultiFab.

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of numpy.array
            A list of NumPy n-dimensional arrays, for each local block in the
            MultiFab.

        """
    def to_xp(self, copy=False, order="F"):
        """

        Provide a NumPy or CuPy view into a MultiFab,
        depending on amr.Config.have_gpu .

        This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
        https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

        This includes ngrow guard cells of each box.

        Note on the order of indices:
        By default, this is as in AMReX in Fortran contiguous order, indexing as
        x,y,z. This has performance implications for use in external libraries such
        as cupy.
        The order="C" option will index as z,y,x and perform better with cupy.
        https://github.com/AMReX-Codes/pyamrex/issues/55#issuecomment-1579610074

        Parameters
        ----------
        self : amrex.MultiFab
            A MultiFab class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).
        order : string, optional
            F order (default) or C. C is faster with external libraries.

        Returns
        -------
        list of xp.array
            A list of NumPy or CuPy n-dimensional arrays, for each local block in the
            MultiFab.

        """
    @property
    def n_comp(self) -> int: ...
    @property
    def n_grow_vect(self) -> IntVect2D: ...
    @property
    def shape(self, include_ghosts=False):
        """
        Returns the shape of the global array

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX
            include_ghosts : bool, default=False
                Whether or not ghost cells are included

        """
    @property
    def shape_with_ghosts(self):
        """
        Returns the shape of the global array including ghost cells

            Parameters
            ----------
            self : amrex.MultiFab
                A MultiFab class in pyAMReX

        """

def AlmostEqual(rb1: RealBox, rb2: RealBox, eps: typing.SupportsFloat = 0.0) -> bool:
    """
    Determine if two boxes are equal to within a tolerance
    """

def EB2_Build(
    geom: Geometry,
    required_coarsening_level: typing.SupportsInt,
    max_coarsening_level: typing.SupportsInt,
    ngrow: typing.SupportsInt = 4,
    build_coarse_level_by_coarsening: bool = True,
    extend_domain_face: bool = True,
    num_coarsen_opt: typing.SupportsInt = 0,
) -> None:
    """
    EB generation
    """

def MPMD_AppNum() -> int: ...
def MPMD_Finalize() -> None: ...
def MPMD_Initialize_without_split(arg0: list) -> None: ...
def MPMD_Initialized() -> bool: ...
def MPMD_MyProc() -> int: ...
def MPMD_MyProgId() -> int: ...
def MPMD_NProcs() -> int: ...
def The_Arena() -> Arena: ...
def The_Async_Arena() -> Arena: ...
def The_Cpu_Arena() -> Arena: ...
def The_Device_Arena() -> Arena: ...
def The_Managed_Arena() -> Arena: ...
def The_Pinned_Arena() -> Arena: ...
def almost_equal(
    x: typing.SupportsFloat, y: typing.SupportsFloat, ulp: typing.SupportsInt = 2
) -> bool: ...
def begin(arg0: Box) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect1D, arg1: IntVect1D) -> IntVect1D: ...
@typing.overload
def coarsen(arg0: Dim3, arg1: IntVect1D) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect1D, arg1: typing.SupportsInt) -> IntVect1D: ...
@typing.overload
def coarsen(arg0: IntVect2D, arg1: IntVect2D) -> IntVect2D: ...
@typing.overload
def coarsen(arg0: Dim3, arg1: IntVect2D) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect2D, arg1: typing.SupportsInt) -> IntVect2D: ...
@typing.overload
def coarsen(arg0: IntVect3D, arg1: IntVect3D) -> IntVect3D: ...
@typing.overload
def coarsen(arg0: Dim3, arg1: IntVect3D) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect3D, arg1: typing.SupportsInt) -> IntVect3D: ...
def concatenate(
    root: str, num: typing.SupportsInt, mindigits: typing.SupportsInt = 5
) -> str:
    """
    Builds plotfile name
    """

@typing.overload
def copy_mfab(
    dst: iMultiFab,
    src: iMultiFab,
    srccomp: typing.SupportsInt,
    dstcomp: typing.SupportsInt,
    numcomp: typing.SupportsInt,
    nghost: typing.SupportsInt,
) -> None: ...
@typing.overload
def copy_mfab(
    dst: iMultiFab,
    src: iMultiFab,
    srccomp: typing.SupportsInt,
    dstcomp: typing.SupportsInt,
    numcomp: typing.SupportsInt,
    nghost: IntVect2D,
) -> None: ...
@typing.overload
def copy_mfab(
    dst: MultiFab,
    src: MultiFab,
    srccomp: typing.SupportsInt,
    dstcomp: typing.SupportsInt,
    numcomp: typing.SupportsInt,
    nghost: typing.SupportsInt,
) -> None: ...
@typing.overload
def copy_mfab(
    dst: MultiFab,
    src: MultiFab,
    srccomp: typing.SupportsInt,
    dstcomp: typing.SupportsInt,
    numcomp: typing.SupportsInt,
    nghost: IntVect2D,
) -> None: ...
@typing.overload
def dtoh_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None:
    """
    Copy from a device to host FabArray.
    """

@typing.overload
def dtoh_memcpy(
    dest: FabArray_FArrayBox,
    src: FabArray_FArrayBox,
    scomp: typing.SupportsInt,
    dcomp: typing.SupportsInt,
    ncomp: typing.SupportsInt,
) -> None:
    """
    Copy from a device to host FabArray for a specific (number of) component(s).
    """

def end(arg0: Box) -> Dim3: ...
@typing.overload
def finalize() -> None: ...
@typing.overload
def finalize(arg0: AMReX) -> None: ...
@typing.overload
def htod_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None:
    """
    Copy from a host to device FabArray.
    """

@typing.overload
def htod_memcpy(
    dest: FabArray_FArrayBox,
    src: FabArray_FArrayBox,
    scomp: typing.SupportsInt,
    dcomp: typing.SupportsInt,
    ncomp: typing.SupportsInt,
) -> None:
    """
    Copy from a host to device FabArray for a specific (number of) component(s).
    """

def initialize(arg0: list) -> AMReX:
    """
    Initialize AMReX library
    """

def initialize_when_MPMD(arg0: list, arg1: typing.Any) -> AMReX: ...
def initialized() -> bool:
    """
    Returns true if there are any currently-active and initialized AMReX instances (i.e. one for which amrex::Initialize has been called, and amrex::Finalize has not). Otherwise false.
    """

def is_valid(arg0: typing.SupportsInt) -> bool: ...
@typing.overload
def lbound(arg0: Box) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_float) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_double) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_longdouble) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_float_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_double_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_longdouble_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_cfloat) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_cdouble) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_cfloat_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_cdouble_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_short) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_int) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_long) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_longlong) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_short_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_int_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_long_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_longlong_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ushort) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_uint) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ulong) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ulonglong) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ushort_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_uint_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ulong_const) -> Dim3: ...
@typing.overload
def lbound(arg0: Array4_ulonglong_const) -> Dim3: ...
@typing.overload
def length(arg0: Box) -> Dim3: ...
@typing.overload
def length(arg0: Array4_float) -> Dim3: ...
@typing.overload
def length(arg0: Array4_double) -> Dim3: ...
@typing.overload
def length(arg0: Array4_longdouble) -> Dim3: ...
@typing.overload
def length(arg0: Array4_float_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_double_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_longdouble_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_cfloat) -> Dim3: ...
@typing.overload
def length(arg0: Array4_cdouble) -> Dim3: ...
@typing.overload
def length(arg0: Array4_cfloat_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_cdouble_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_short) -> Dim3: ...
@typing.overload
def length(arg0: Array4_int) -> Dim3: ...
@typing.overload
def length(arg0: Array4_long) -> Dim3: ...
@typing.overload
def length(arg0: Array4_longlong) -> Dim3: ...
@typing.overload
def length(arg0: Array4_short_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_int_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_long_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_longlong_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ushort) -> Dim3: ...
@typing.overload
def length(arg0: Array4_uint) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ulong) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ulonglong) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ushort_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_uint_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ulong_const) -> Dim3: ...
@typing.overload
def length(arg0: Array4_ulonglong_const) -> Dim3: ...
def makeEBFabFactory(
    geom: Geometry,
    ba: BoxArray,
    dm: DistributionMapping,
    ngrow: Vector_int,
    support: EBSupport,
) -> EBFArrayBoxFactory:
    """
    Make EBFArrayBoxFactory for given Geometry, BoxArray and DistributionMapping
    """

def make_invalid(arg0: typing.SupportsInt) -> int: ...
def make_valid(arg0: typing.SupportsInt) -> int: ...
def max(arg0: RealVect, arg1: RealVect) -> RealVect: ...
def min(arg0: RealVect, arg1: RealVect) -> RealVect: ...
def pack_cpus(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64],
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
) -> typing.Any: ...
def pack_ids(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64],
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int64],
) -> typing.Any: ...
@typing.overload
def refine(arg0: Dim3, arg1: IntVect1D) -> Dim3: ...
@typing.overload
def refine(arg0: Dim3, arg1: IntVect2D) -> Dim3: ...
@typing.overload
def refine(arg0: Dim3, arg1: IntVect3D) -> Dim3: ...
def size() -> int:
    """
    The amr stack size, the number of amr instances pushed.
    """

@typing.overload
def ubound(arg0: Box) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_float) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_double) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_longdouble) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_float_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_double_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_longdouble_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_cfloat) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_cdouble) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_cfloat_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_cdouble_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_short) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_int) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_long) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_longlong) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_short_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_int_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_long_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_longlong_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ushort) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_uint) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ulong) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ulonglong) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ushort_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_uint_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ulong_const) -> Dim3: ...
@typing.overload
def ubound(arg0: Array4_ulonglong_const) -> Dim3: ...
def unpack_cpus(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64],
) -> typing.Any: ...
def unpack_ids(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64],
) -> typing.Any: ...
def write_single_level_plotfile(
    plotfilename: str,
    mf: MultiFab,
    varnames: Vector_string,
    geom: Geometry,
    time: typing.SupportsFloat,
    level_step: typing.SupportsInt,
    versionName: str = "HyperCLaw-V1.1",
    levelPrefix: str = "Level_",
    mfPrefix: str = "Cell",
    extra_dirs: Vector_string = ...,
) -> None:
    """
    Writes single level plotfile
    """

Exact: GrowthStrategy  # value = <GrowthStrategy.Exact: 1>
Geometric: GrowthStrategy  # value = <GrowthStrategy.Geometric: 2>
Poisson: GrowthStrategy  # value = <GrowthStrategy.Poisson: 0>
__author__: str = "Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang"
__license__: str = "BSD-3-Clause-LBNL"
__version__: str = "25.11-15-g4dad1664d746"
basic: EBSupport  # value = <EBSupport.basic: 1>
full: EBSupport  # value = <EBSupport.full: 3>
volume: EBSupport  # value = <EBSupport.volume: 2>
IntVect = IntVect2D
