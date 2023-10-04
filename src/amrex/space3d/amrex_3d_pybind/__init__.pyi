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

import typing

import numpy

from . import ParallelDescriptor

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
    "ArrayOfStructs_0_0_arena",
    "ArrayOfStructs_0_0_default",
    "ArrayOfStructs_0_0_pinned",
    "ArrayOfStructs_1_1_arena",
    "ArrayOfStructs_1_1_default",
    "ArrayOfStructs_1_1_pinned",
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
    "Geometry",
    "GeometryData",
    "IndexType",
    "IntVect",
    "MFInfo",
    "MFIter",
    "MultiFab",
    "PIdx",
    "PODVector_int_arena",
    "PODVector_int_pinned",
    "PODVector_int_std",
    "PODVector_real_arena",
    "PODVector_real_pinned",
    "PODVector_real_std",
    "ParConstIterBase_0_0_4_0_arena",
    "ParConstIterBase_0_0_4_0_default",
    "ParConstIterBase_0_0_4_0_pinned",
    "ParConstIterBase_0_0_5_0_arena",
    "ParConstIterBase_0_0_5_0_default",
    "ParConstIterBase_0_0_5_0_pinned",
    "ParConstIterBase_1_1_2_1_arena",
    "ParConstIterBase_1_1_2_1_default",
    "ParConstIterBase_1_1_2_1_pinned",
    "ParConstIterBase_pureSoA_8_2_arena",
    "ParConstIterBase_pureSoA_8_2_default",
    "ParConstIterBase_pureSoA_8_2_pinned",
    "ParConstIter_0_0_4_0_arena",
    "ParConstIter_0_0_4_0_default",
    "ParConstIter_0_0_4_0_pinned",
    "ParConstIter_0_0_5_0_arena",
    "ParConstIter_0_0_5_0_default",
    "ParConstIter_0_0_5_0_pinned",
    "ParConstIter_1_1_2_1_arena",
    "ParConstIter_1_1_2_1_default",
    "ParConstIter_1_1_2_1_pinned",
    "ParConstIter_pureSoA_8_2_arena",
    "ParConstIter_pureSoA_8_2_default",
    "ParConstIter_pureSoA_8_2_pinned",
    "ParIterBase_0_0_4_0_arena",
    "ParIterBase_0_0_4_0_default",
    "ParIterBase_0_0_4_0_pinned",
    "ParIterBase_0_0_5_0_arena",
    "ParIterBase_0_0_5_0_default",
    "ParIterBase_0_0_5_0_pinned",
    "ParIterBase_1_1_2_1_arena",
    "ParIterBase_1_1_2_1_default",
    "ParIterBase_1_1_2_1_pinned",
    "ParIterBase_pureSoA_8_2_arena",
    "ParIterBase_pureSoA_8_2_default",
    "ParIterBase_pureSoA_8_2_pinned",
    "ParIter_0_0_4_0_arena",
    "ParIter_0_0_4_0_default",
    "ParIter_0_0_4_0_pinned",
    "ParIter_0_0_5_0_arena",
    "ParIter_0_0_5_0_default",
    "ParIter_0_0_5_0_pinned",
    "ParIter_1_1_2_1_arena",
    "ParIter_1_1_2_1_default",
    "ParIter_1_1_2_1_pinned",
    "ParIter_pureSoA_8_2_arena",
    "ParIter_pureSoA_8_2_default",
    "ParIter_pureSoA_8_2_pinned",
    "ParallelDescriptor",
    "ParmParse",
    "ParticleContainer_0_0_4_0_arena",
    "ParticleContainer_0_0_4_0_default",
    "ParticleContainer_0_0_4_0_pinned",
    "ParticleContainer_0_0_5_0_arena",
    "ParticleContainer_0_0_5_0_default",
    "ParticleContainer_0_0_5_0_pinned",
    "ParticleContainer_1_1_2_1_arena",
    "ParticleContainer_1_1_2_1_default",
    "ParticleContainer_1_1_2_1_pinned",
    "ParticleContainer_pureSoA_8_2_arena",
    "ParticleContainer_pureSoA_8_2_default",
    "ParticleContainer_pureSoA_8_2_pinned",
    "ParticleInitType_0_0_4_0",
    "ParticleInitType_0_0_5_0",
    "ParticleInitType_1_1_2_1",
    "ParticleTileData_0_0_37_1",
    "ParticleTileData_0_0_4_0",
    "ParticleTileData_0_0_5_0",
    "ParticleTileData_0_0_8_2",
    "ParticleTileData_1_1_2_1",
    "ParticleTile_0_0_37_1_arena",
    "ParticleTile_0_0_37_1_default",
    "ParticleTile_0_0_37_1_pinned",
    "ParticleTile_0_0_4_0_arena",
    "ParticleTile_0_0_4_0_default",
    "ParticleTile_0_0_4_0_pinned",
    "ParticleTile_0_0_5_0_arena",
    "ParticleTile_0_0_5_0_default",
    "ParticleTile_0_0_5_0_pinned",
    "ParticleTile_0_0_8_2_arena",
    "ParticleTile_0_0_8_2_default",
    "ParticleTile_0_0_8_2_pinned",
    "ParticleTile_1_1_2_1_arena",
    "ParticleTile_1_1_2_1_default",
    "ParticleTile_1_1_2_1_pinned",
    "Particle_0_0",
    "Particle_1_1",
    "Particle_2_1",
    "Particle_37_1",
    "Particle_3_2",
    "Particle_4_0",
    "Particle_5_0",
    "Particle_7_0",
    "Particle_8_2",
    "Periodicity",
    "RealBox",
    "RealVect",
    "StructOfArrays_2_1_arena",
    "StructOfArrays_2_1_default",
    "StructOfArrays_2_1_pinned",
    "StructOfArrays_37_1_arena",
    "StructOfArrays_37_1_default",
    "StructOfArrays_37_1_pinned",
    "StructOfArrays_4_0_arena",
    "StructOfArrays_4_0_default",
    "StructOfArrays_4_0_pinned",
    "StructOfArrays_5_0_arena",
    "StructOfArrays_5_0_default",
    "StructOfArrays_5_0_pinned",
    "StructOfArrays_8_2_arena",
    "StructOfArrays_8_2_default",
    "StructOfArrays_8_2_pinned",
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
    "begin",
    "coarsen",
    "concatenate",
    "dtoh_memcpy",
    "end",
    "finalize",
    "htod_memcpy",
    "initialize",
    "initialized",
    "lbound",
    "length",
    "max",
    "min",
    "refine",
    "size",
    "ubound",
    "unpack_cpus",
    "unpack_ids",
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
    grid_eff: float
    iterate_on_new_grids: bool
    max_level: int
    n_proper: int
    refine_grid_layout: bool
    refine_grid_layout_dims: IntVect
    use_fixed_coarse_grids: bool
    use_fixed_upto_level: int
    use_new_chop: bool
    verbose: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def blocking_factor(self, arg0: int) -> IntVect: ...
    def max_grid_size(self, arg0: int) -> IntVect: ...
    def n_error_buf(self, arg0: int) -> IntVect: ...
    def ref_ratio(self, arg0: int) -> IntVect: ...

class AmrMesh:
    def Verbose(self) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        rb: RealBox,
        max_level_in: int,
        n_cell_in: Vector_int,
        coord: int,
        ref_ratios: Vector_IntVect,
        is_per: list[int[3]],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def finest_level(self) -> int: ...
    def max_level(self) -> int: ...
    @typing.overload
    def ref_ratio(self) -> Vector_IntVect: ...
    @typing.overload
    def ref_ratio(self, arg0: int) -> IntVect: ...

class Arena:
    pass

class Array4_cdouble:
    @typing.overload
    def __getitem__(self, arg0: IntVect) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.complex128]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: complex) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.complex128]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cdouble_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[complex]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.complex128]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.complex64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: complex) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: complex) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.complex64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> complex: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> complex: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_cfloat_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[complex]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.complex64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: float) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.float64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_double_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.float64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float32]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: float) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.float32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_float_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float32]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.float32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_int_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_long_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int64]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.longdouble]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: float) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.longdouble]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longdouble_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.longdouble]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.longdouble]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_longlong_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int64]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int16]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_short_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int16]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.int16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint32]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_uint_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint32]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint32]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulong_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint64]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint64]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ulonglong_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint64]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint64]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint16]) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: IntVect, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[4]], arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: list[int[3]], arg1: int) -> None: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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
    def __getitem__(self, arg0: IntVect) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[4]]) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: list[int[3]]) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort_const) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort_const, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: Array4_ushort_const, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.uint16]) -> None: ...
    def __repr__(self) -> str: ...
    def contains(self, arg0: int, arg1: int, arg2: int) -> bool: ...
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into an Array4.

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
    def to_host(self) -> numpy.ndarray[numpy.uint16]: ...
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into an Array4.

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
            A numpy n-dimensional array.

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

class ArrayOfStructs_0_0_arena:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_0_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_0_0) -> None: ...
    def back(self) -> Particle_0_0:
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
    def push_back(self, arg0: Particle_0_0) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_0_0_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_0_0_default:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_0_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_0_0) -> None: ...
    def back(self) -> Particle_0_0:
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
    def push_back(self, arg0: Particle_0_0) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_0_0_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_0_0_pinned:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_0_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_0_0) -> None: ...
    def back(self) -> Particle_0_0:
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
    def push_back(self, arg0: Particle_0_0) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_0_0_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_1_1_arena:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_1_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_1_1) -> None: ...
    def back(self) -> Particle_1_1:
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
    def push_back(self, arg0: Particle_1_1) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_1_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_1_1_default:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_1_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_1_1) -> None: ...
    def back(self) -> Particle_1_1:
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
    def push_back(self, arg0: Particle_1_1) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_1_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_1_1_pinned:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_1_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_1_1) -> None: ...
    def back(self) -> Particle_1_1:
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
    def push_back(self, arg0: Particle_1_1) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_1_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_arena:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_2_1) -> None: ...
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
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_default:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_2_1) -> None: ...
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
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ArrayOfStructs_2_1_pinned:
    @staticmethod
    def test_sizes() -> None: ...
    def __getitem__(self, arg0: int) -> Particle_2_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_2_1) -> None: ...
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
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_host(self) -> ArrayOfStructs_2_1_default: ...
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a ArrayOfStructs.

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
            of 1D numpy arrays.

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
    def __init__(self, arg0: Box, arg1: int, arg2: Arena) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int, arg2: float) -> None: ...
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
    def big_end(self) -> IntVect: ...
    def box(self) -> Box: ...
    def clear(self) -> None: ...
    def const_array(self) -> Array4_double_const: ...
    def hi_vect(self) -> int: ...
    def is_allocated(self) -> bool: ...
    def length(self) -> IntVect: ...
    def lo_vect(self) -> int: ...
    @typing.overload
    def n_bytes(self) -> int: ...
    @typing.overload
    def n_bytes(self, arg0: Box, arg1: int) -> int: ...
    def n_bytes_owned(self) -> int: ...
    def n_comp(self) -> int: ...
    def num_pts(self) -> int: ...
    def resize(self, arg0: Box, arg1: int, arg2: Arena) -> None: ...
    def size(self) -> int: ...
    def small_end(self) -> IntVect: ...
    def to_host(self) -> BaseFab_Real: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class Box:
    def __add__(self, arg0: IntVect) -> Box: ...
    def __iadd__(self, arg0: IntVect) -> Box: ...
    @typing.overload
    def __init__(self, small: IntVect, big: IntVect) -> None: ...
    @typing.overload
    def __init__(self, small: IntVect, big: IntVect, typ: IntVect) -> None: ...
    @typing.overload
    def __init__(self, small: IntVect, big: IntVect, t: IndexType) -> None: ...
    @typing.overload
    def __init__(self, small: list[int[3]], big: list[int[3]]) -> None: ...
    @typing.overload
    def __init__(
        self, small: list[int[3]], big: list[int[3]], t: IndexType
    ) -> None: ...
    def __isub__(self, arg0: IntVect) -> Box: ...
    def __iter__(self) -> typing.Iterator: ...
    def __repr__(self) -> str: ...
    def __sub__(self, arg0: IntVect) -> Box: ...
    def begin(self, arg0: Box) -> Dim3: ...
    def contains(self, arg0: IntVect) -> bool: ...
    @typing.overload
    def convert(self, arg0: IndexType) -> Box: ...
    @typing.overload
    def convert(self, arg0: IntVect) -> Box: ...
    @typing.overload
    def enclosed_cells(self) -> Box: ...
    @typing.overload
    def enclosed_cells(self, dir: int) -> Box: ...
    @typing.overload
    def enclosed_cells(self, d: Direction) -> Box: ...
    def end(self, arg0: Box) -> Dim3: ...
    @typing.overload
    def grow(self, n_cell: int) -> Box: ...
    @typing.overload
    def grow(self, n_cells: IntVect) -> Box: ...
    @typing.overload
    def grow(self, idir: int, n_cell: int) -> Box: ...
    @typing.overload
    def grow(self, d: Direction, n_cell: int) -> Box: ...
    def intersects(self, arg0: Box) -> bool: ...
    def lbound(self, arg0: Box) -> Dim3: ...
    @typing.overload
    def length(self) -> IntVect:
        """
        Return IntVect of lengths of the Box
        """
    @typing.overload
    def length(self, arg0: int) -> int:
        """
        Return the length of the Box in given direction.
        """
    def make_slab(self, direction: int, slab_index: int) -> Box: ...
    def normalize(self) -> None: ...
    def numPts(self) -> int:
        """
        Return the number of points in the Box.
        """
    def same_size(self, arg0: Box) -> bool: ...
    def same_type(self, arg0: Box) -> bool: ...
    def shift(self, arg0: IntVect) -> Box: ...
    def strictly_contains(self, arg0: IntVect) -> bool: ...
    @typing.overload
    def surrounding_nodes(self) -> Box: ...
    @typing.overload
    def surrounding_nodes(self, dir: int) -> Box: ...
    @typing.overload
    def surrounding_nodes(self, d: Direction) -> Box: ...
    def ubound(self, arg0: Box) -> Dim3: ...
    @property
    def big_end(self) -> IntVect: ...
    @property
    def cell_centered(self) -> bool: ...
    @property
    def d_num_pts(self) -> float: ...
    @property
    def hi_vect(self) -> IntVect: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def is_square(self) -> bool: ...
    @property
    def ix_type(self) -> IndexType: ...
    @property
    def lo_vect(self) -> IntVect: ...
    @property
    def num_pts(self) -> int: ...
    @property
    def ok(self) -> bool: ...
    @property
    def size(self) -> IntVect: ...
    @property
    def small_end(self) -> IntVect: ...
    @property
    def the_unit_box() -> Box: ...
    @property
    def type(self) -> IntVect: ...
    @type.setter
    def type(self, arg1: IndexType) -> Box: ...
    @property
    def volume(self) -> int: ...

class BoxArray:
    def __getitem__(self, arg0: int) -> Box: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int) -> None: ...
    def __repr__(self) -> str: ...
    def cell_equal(self, arg0: BoxArray) -> bool: ...
    def clear(self) -> None: ...
    @typing.overload
    def coarsen(self, arg0: IntVect) -> BoxArray: ...
    @typing.overload
    def coarsen(self, arg0: int) -> BoxArray: ...
    @typing.overload
    def coarsenable(self, arg0: int, arg1: int) -> bool: ...
    @typing.overload
    def coarsenable(self, arg0: IntVect, arg1: int) -> bool: ...
    @typing.overload
    def coarsenable(self, arg0: IntVect, arg1: IntVect) -> bool: ...
    def define(self, arg0: Box) -> None: ...
    def get(self, arg0: int) -> Box: ...
    def ix_type(self) -> IndexType: ...
    @typing.overload
    def max_size(self, arg0: int) -> BoxArray: ...
    @typing.overload
    def max_size(self, arg0: IntVect) -> BoxArray: ...
    def minimal_box(self) -> Box: ...
    @typing.overload
    def refine(self, arg0: int) -> BoxArray: ...
    @typing.overload
    def refine(self, arg0: IntVect) -> BoxArray: ...
    def resize(self, arg0: int) -> None: ...
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
    amrex_version: typing.ClassVar[str] = "23.09"
    gpu_backend = None
    have_gpu: typing.ClassVar[bool] = False
    have_mpi: typing.ClassVar[bool] = True
    have_omp: typing.ClassVar[bool] = False
    spacedim: typing.ClassVar[int] = 3
    verbose: typing.ClassVar[int] = 1

class CoordSys:
    class CoordType:
        """
        Members:

          undef

          cartesian

          RZ

          SPHERICAL
        """

        RZ: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.RZ: 1>
        SPHERICAL: typing.ClassVar[
            CoordSys.CoordType
        ]  # value = <CoordType.SPHERICAL: 2>
        __members__: typing.ClassVar[
            dict[str, CoordSys.CoordType]
        ]  # value = {'undef': <CoordType.undef: -1>, 'cartesian': <CoordType.cartesian: 0>, 'RZ': <CoordType.RZ: 1>, 'SPHERICAL': <CoordType.SPHERICAL: 2>}
        cartesian: typing.ClassVar[
            CoordSys.CoordType
        ]  # value = <CoordType.cartesian: 0>
        undef: typing.ClassVar[CoordSys.CoordType]  # value = <CoordType.undef: -1>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: int) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: int) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...
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
    x: int
    y: int
    z: int
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Direction:
    pass

class DistributionMapping:
    def ProcessorMap(self) -> Vector_int: ...
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: DistributionMapping) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None: ...
    @typing.overload
    def __init__(self, boxes: BoxArray) -> None: ...
    @typing.overload
    def __init__(self, boxes: BoxArray, nprocs: int) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def define(self, boxes: BoxArray) -> None: ...
    @typing.overload
    def define(self, boxes: BoxArray, nprocs: int) -> None: ...
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

class FArrayBox(BaseFab_Real):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Arena) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int, arg2: Arena) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: Box, arg1: int, arg2: bool, arg3: bool, arg4: Arena
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: Box, arg1: int, arg2: float) -> None: ...
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
    def __iter__(self) -> MFIter: ...
    def is_nodal(self, arg0: int) -> bool: ...
    @property
    def is_all_cell_centered(self) -> bool: ...
    @property
    def is_all_nodal(self) -> bool: ...
    @property
    def nComp(self) -> int: ...
    @property
    def nGrowVect(self) -> IntVect: ...
    @property
    def num_comp(self) -> int: ...
    @property
    def size(self) -> int: ...

class FabArray_FArrayBox(FabArrayBase):
    @staticmethod
    def lin_comb(
        arg0: FabArray_FArrayBox,
        arg1: float,
        arg2: FabArray_FArrayBox,
        arg3: int,
        arg4: float,
        arg5: FabArray_FArrayBox,
        arg6: int,
        arg7: int,
        arg8: int,
        arg9: IntVect,
    ) -> None: ...
    @staticmethod
    def saxpy(
        arg0: FabArray_FArrayBox,
        arg1: float,
        arg2: FabArray_FArrayBox,
        arg3: int,
        arg4: int,
        arg5: int,
        arg6: IntVect,
    ) -> None: ...
    @staticmethod
    def xpay(
        arg0: FabArray_FArrayBox,
        arg1: float,
        arg2: FabArray_FArrayBox,
        arg3: int,
        arg4: int,
        arg5: int,
        arg6: IntVect,
    ) -> None: ...
    def array(self, arg0: MFIter) -> Array4_double: ...
    def const_array(self, arg0: MFIter) -> Array4_double_const: ...
    @typing.overload
    def fill_boundary(self, cross: bool = False) -> None: ...
    @typing.overload
    def fill_boundary(self, period: Periodicity, cross: bool = False) -> None: ...
    @typing.overload
    def fill_boundary(
        self, nghost: IntVect, period: Periodicity, cross: bool = False
    ) -> None: ...
    @typing.overload
    def fill_boundary(self, scomp: int, ncomp: int, cross: bool = False) -> None: ...
    @typing.overload
    def fill_boundary(
        self, scomp: int, ncomp: int, period: Periodicity, cross: bool = False
    ) -> None: ...
    @typing.overload
    def fill_boundary(
        self,
        scomp: int,
        ncomp: int,
        nghost: IntVect,
        period: Periodicity,
        cross: bool = False,
    ) -> None: ...
    def override_sync(self, arg0: Periodicity) -> None: ...
    def sum(self, arg0: int, arg1: IntVect, arg2: bool) -> float: ...
    @typing.overload
    def sum_boundary(self, arg0: Periodicity) -> None: ...
    @typing.overload
    def sum_boundary(self, arg0: int, arg1: int, arg2: Periodicity) -> None: ...
    @typing.overload
    def sum_boundary(
        self, arg0: int, arg1: int, arg2: IntVect, arg3: Periodicity
    ) -> None: ...

class Geometry(CoordSys):
    @typing.overload
    def Domain(self) -> Box:
        """
        Return rectangular domain
        """
    @typing.overload
    def Domain(self, arg0: Box) -> None: ...
    @typing.overload
    def ProbDomain(self) -> RealBox:
        """
        Return problem domain
        """
    @typing.overload
    def ProbDomain(self, arg0: RealBox) -> None: ...
    @typing.overload
    def ProbHi(self, arg0: int) -> float:
        """
        Get the hi end of the problem domain in specified direction
        """
    @typing.overload
    def ProbHi(self) -> list[float[3]]:
        """
        Get the list of lo ends of the problem domain
        """
    def ProbLength(self, arg0: int) -> float:
        """
        length of problem domain in specified dimension
        """
    @typing.overload
    def ProbLo(self, arg0: int) -> float:
        """
        Get the lo end of the problem domain in specified direction
        """
    @typing.overload
    def ProbLo(self) -> list[float[3]]:
        """
        Get the list of lo ends of the problem domain
        """
    def ProbSize(self) -> float:
        """
        the overall size of the domain
        """
    def ResetDefaultCoord(self) -> None:
        """
        Reset default coord of Geometry class with an Array of `int`
        """
    def ResetDefaultPeriodicity(self) -> None:
        """
        Reset default periodicity of Geometry class with an Array of `int`
        """
    def ResetDefaultProbDomain(self) -> None:
        """
        Reset default problem domain of Geometry class with a `RealBox`
        """
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, dom: Box, rb: RealBox, coord: int, is_per: list[int[3]]
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def coarsen(self, arg0: IntVect) -> None: ...
    def data(self) -> GeometryData:
        """
        Returns non-static copy of geometry's stored data
        """
    def define(self, arg0: Box, arg1: RealBox, arg2: int, arg3: list[int[3]]) -> None:
        """
        Set geometry
        """
    @typing.overload
    def growNonPeriodicDomain(self, arg0: IntVect) -> Box: ...
    @typing.overload
    def growNonPeriodicDomain(self, arg0: int) -> Box: ...
    @typing.overload
    def growPeriodicDomain(self, arg0: IntVect) -> Box: ...
    @typing.overload
    def growPeriodicDomain(self, arg0: int) -> Box: ...
    def insideRoundOffDomain(self, arg0: float, arg1: float, arg2: float) -> bool:
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
    def isPeriodic(self, arg0: int) -> bool:
        """
        Is the domain periodic in the specified direction?
        """
    @typing.overload
    def isPeriodic(self) -> list[int[3]]:
        """
        Return list indicating whether domain is periodic in each direction
        """
    def outsideRoundOffDomain(self, arg0: float, arg1: float, arg2: float) -> bool:
        """
        Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()
        """
    def period(self, arg0: int) -> int:
        """
        Return the period in the specified direction
        """
    @typing.overload
    def periodicity(self) -> Periodicity: ...
    @typing.overload
    def periodicity(self, arg0: Box) -> Periodicity:
        """
        Return Periodicity object with lengths determined by input Box
        """
    def refine(self, arg0: IntVect) -> None: ...
    def setPeriodicity(self, arg0: list[int[3]]) -> list[int[3]]: ...

class GeometryData:
    @typing.overload
    def CellSize(self) -> list[float[3]]:
        """
        Returns the cellsize for each coordinate direction.
        """
    @typing.overload
    def CellSize(self, arg0: int) -> float:
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
    def ProbHi(self) -> list[float[3]]:
        """
        Returns the hi end for each coordinate direction.
        """
    @typing.overload
    def ProbHi(self, arg0: int) -> float:
        """
        Returns the hi end of the problem domain in specified dimension.
        """
    @typing.overload
    def ProbLo(self) -> list[float[3]]:
        """
        Returns the lo end for each coordinate direction.
        """
    @typing.overload
    def ProbLo(self, arg0: int) -> float:
        """
        Returns the lo end of the problem domain in specified dimension.
        """
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def isPeriodic(self) -> list[int[3]]:
        """
        Returns whether the domain is periodic in each direction.
        """
    @typing.overload
    def isPeriodic(self, arg0: int) -> int:
        """
        Returns whether the domain is periodic in the given direction.
        """
    @property
    def coord(self) -> int: ...
    @property
    def domain(self) -> Box: ...
    @property
    def dx(self) -> list[float[3]]: ...
    @property
    def is_periodic(self) -> list[int[3]]: ...
    @property
    def prob_domain(self) -> RealBox: ...

class IndexType:
    class CellIndex:
        """
        Members:

          CELL

          NODE
        """

        CELL: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.CELL: 0>
        NODE: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.NODE: 1>
        __members__: typing.ClassVar[
            dict[str, IndexType.CellIndex]
        ]  # value = {'CELL': <CellIndex.CELL: 0>, 'NODE': <CellIndex.NODE: 1>}
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: int) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: int) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...
    CELL: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.CELL: 0>
    NODE: typing.ClassVar[IndexType.CellIndex]  # value = <CellIndex.NODE: 1>
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_type() -> IndexType: ...
    @staticmethod
    def node_type() -> IndexType: ...
    def __eq__(self, arg0: IndexType) -> bool: ...
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: IndexType) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: IndexType.CellIndex,
        arg1: IndexType.CellIndex,
        arg2: IndexType.CellIndex,
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
    def cell_centered(self, arg0: int) -> bool: ...
    def clear(self) -> None: ...
    def flip(self, arg0: int) -> None: ...
    @typing.overload
    def ix_type(self) -> IntVect: ...
    @typing.overload
    def ix_type(self, arg0: int) -> IndexType.CellIndex: ...
    @typing.overload
    def node_centered(self) -> bool: ...
    @typing.overload
    def node_centered(self, arg0: int) -> bool: ...
    def ok(self) -> bool: ...
    def set(self, arg0: int) -> None: ...
    def set_type(self, arg0: int, arg1: IndexType.CellIndex) -> None: ...
    def setall(self) -> None: ...
    def test(self, arg0: int) -> bool: ...
    def to_IntVect(self) -> IntVect: ...
    def unset(self, arg0: int) -> None: ...

class IntVect:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def cell_vector() -> IntVect: ...
    @staticmethod
    def max_vector() -> IntVect: ...
    @staticmethod
    def min_vector() -> IntVect: ...
    @staticmethod
    def node_vector() -> IntVect: ...
    @staticmethod
    def unit_vector() -> IntVect: ...
    @staticmethod
    def zero_vector() -> IntVect: ...
    @typing.overload
    def __add__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __add__(self, arg0: IntVect) -> IntVect: ...
    @typing.overload
    def __eq__(self, arg0: int) -> bool: ...
    @typing.overload
    def __eq__(self, arg0: IntVect) -> bool: ...
    def __ge__(self, arg0: IntVect) -> bool: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __gt__(self, arg0: IntVect) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __iadd__(self, arg0: IntVect) -> IntVect: ...
    @typing.overload
    def __imul__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __imul__(self, arg0: IntVect) -> IntVect: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: list[int[3]]) -> None: ...
    @typing.overload
    def __isub__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __isub__(self, arg0: IntVect) -> IntVect: ...
    def __iter__(self) -> typing.Iterator: ...
    @typing.overload
    def __itruediv__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __itruediv__(self, arg0: IntVect) -> IntVect: ...
    def __le__(self, arg0: IntVect) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, arg0: IntVect) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __mul__(self, arg0: IntVect) -> IntVect: ...
    @typing.overload
    def __ne__(self, arg0: int) -> bool: ...
    @typing.overload
    def __ne__(self, arg0: IntVect) -> bool: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: int) -> int: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __sub__(self, arg0: IntVect) -> IntVect: ...
    @typing.overload
    def __truediv__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __truediv__(self, arg0: IntVect) -> IntVect: ...
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

class MFIter:
    @typing.overload
    def __init__(self, arg0: FabArrayBase) -> None: ...
    @typing.overload
    def __init__(self, arg0: MultiFab) -> None: ...
    def __next__(self) -> MFIter: ...
    def __repr__(self) -> str: ...
    def fabbox(self) -> Box: ...
    @typing.overload
    def grownnodaltilebox(self, int: int = -1, ng: int = -1000000) -> Box: ...
    @typing.overload
    def grownnodaltilebox(self, int: int, ng: IntVect) -> Box: ...
    def growntilebox(self, ng: IntVect = -1000000) -> Box: ...
    def nodaltilebox(self, dir: int = -1) -> Box: ...
    @typing.overload
    def tilebox(self) -> Box: ...
    @typing.overload
    def tilebox(self, arg0: IntVect) -> Box: ...
    @typing.overload
    def tilebox(self, arg0: IntVect, arg1: IntVect) -> Box: ...
    def validbox(self) -> Box: ...
    @property
    def index(self) -> int: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def length(self) -> int: ...

class MultiFab(FabArray_FArrayBox):
    @staticmethod
    @typing.overload
    def add(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: int
    ) -> None: ...
    @staticmethod
    @typing.overload
    def add(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: IntVect
    ) -> None: ...
    @staticmethod
    @typing.overload
    def add_product(
        arg0: MultiFab,
        arg1: MultiFab,
        arg2: int,
        arg3: MultiFab,
        arg4: int,
        arg5: int,
        arg6: int,
        arg7: int,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def add_product(
        arg0: MultiFab,
        arg1: MultiFab,
        arg2: int,
        arg3: MultiFab,
        arg4: int,
        arg5: int,
        arg6: int,
        arg7: IntVect,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def copy(
        dst: MultiFab,
        src: MultiFab,
        srccomp: int,
        dstcomp: int,
        numcomp: int,
        nghost: int,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def copy(
        dst: MultiFab,
        src: MultiFab,
        srccomp: int,
        dstcomp: int,
        numcomp: int,
        nghost: IntVect,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def divide(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: int
    ) -> None: ...
    @staticmethod
    @typing.overload
    def divide(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: IntVect
    ) -> None: ...
    @staticmethod
    @typing.overload
    def dot(
        arg0: MultiFab,
        arg1: int,
        arg2: MultiFab,
        arg3: int,
        arg4: int,
        arg5: int,
        arg6: bool,
    ) -> float: ...
    @staticmethod
    @typing.overload
    def dot(arg0: MultiFab, arg1: int, arg2: int, arg3: int, arg4: bool) -> float: ...
    @staticmethod
    def finalize() -> None: ...
    @staticmethod
    def initialize() -> None: ...
    @staticmethod
    def lin_comb(
        arg0: MultiFab,
        arg1: float,
        arg2: MultiFab,
        arg3: int,
        arg4: float,
        arg5: MultiFab,
        arg6: int,
        arg7: int,
        arg8: int,
        arg9: int,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def multiply(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: int
    ) -> None: ...
    @staticmethod
    @typing.overload
    def multiply(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: IntVect
    ) -> None: ...
    @staticmethod
    def saxpy(
        arg0: MultiFab,
        arg1: float,
        arg2: MultiFab,
        arg3: int,
        arg4: int,
        arg5: int,
        arg6: int,
    ) -> None: ...
    @staticmethod
    @typing.overload
    def subtract(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: int
    ) -> None: ...
    @staticmethod
    @typing.overload
    def subtract(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: IntVect
    ) -> None: ...
    @staticmethod
    @typing.overload
    def swap(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: int
    ) -> None: ...
    @staticmethod
    @typing.overload
    def swap(
        arg0: MultiFab, arg1: MultiFab, arg2: int, arg3: int, arg4: int, arg5: IntVect
    ) -> None: ...
    @staticmethod
    def xpay(
        arg0: MultiFab,
        arg1: float,
        arg2: MultiFab,
        arg3: int,
        arg4: int,
        arg5: int,
        arg6: int,
    ) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: BoxArray, arg1: DistributionMapping, arg2: int, arg3: int
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: BoxArray,
        arg1: DistributionMapping,
        arg2: int,
        arg3: int,
        arg4: MFInfo,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, arg0: BoxArray, arg1: DistributionMapping, arg2: int, arg3: IntVect
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: BoxArray,
        arg1: DistributionMapping,
        arg2: int,
        arg3: IntVect,
        arg4: MFInfo,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def abs(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def abs(self, arg0: int, arg1: int, arg2: int) -> None: ...
    def average_sync(self, arg0: Periodicity) -> None: ...
    def box_array(self) -> BoxArray: ...
    @typing.overload
    def contains_inf(self, arg0: bool) -> bool: ...
    @typing.overload
    def contains_inf(self, arg0: int, arg1: int, arg2: int, arg3: bool) -> bool: ...
    @typing.overload
    def contains_inf(self, arg0: int, arg1: int, arg2: IntVect, arg3: bool) -> bool: ...
    @typing.overload
    def contains_nan(self, arg0: bool) -> bool: ...
    @typing.overload
    def contains_nan(self, arg0: int, arg1: int, arg2: int, arg3: bool) -> bool: ...
    @typing.overload
    def contains_nan(self, arg0: int, arg1: int, arg2: IntVect, arg3: bool) -> bool: ...
    def divi(self, arg0: MultiFab, arg1: int, arg2: int, arg3: int) -> None: ...
    def dm(self) -> DistributionMapping: ...
    @typing.overload
    def invert(self, arg0: float, arg1: int) -> None: ...
    @typing.overload
    def invert(self, arg0: float, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def invert(self, arg0: float, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def invert(self, arg0: float, arg1: Box, arg2: int) -> None: ...
    @typing.overload
    def invert(
        self, arg0: float, arg1: Box, arg2: int, arg3: int, arg4: int
    ) -> None: ...
    @typing.overload
    def max(self, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the maximum value of the specfied component of the MultiFab.
        """
    @typing.overload
    def max(
        self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False
    ) -> float:
        """
        Returns the maximum value of the specfied component of the MultiFab over the region.
        """
    def maxIndex(self, arg0: int, arg1: int) -> IntVect: ...
    @typing.overload
    def min(self, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the minimum value of the specfied component of the MultiFab.
        """
    @typing.overload
    def min(
        self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False
    ) -> float:
        """
        Returns the minimum value of the specfied component of the MultiFab over the region.
        """
    def minIndex(self, arg0: int, arg1: int) -> IntVect: ...
    def minus(self, arg0: MultiFab, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def mult(self, arg0: float, arg1: int) -> None: ...
    @typing.overload
    def mult(self, arg0: float, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def mult(self, arg0: float, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def mult(self, arg0: float, arg1: Box, arg2: int) -> None: ...
    @typing.overload
    def mult(self, arg0: float, arg1: Box, arg2: int, arg3: int, arg4: int) -> None: ...
    def n_comp(self) -> int: ...
    def n_grow_vect(self) -> IntVect: ...
    @typing.overload
    def negate(self, arg0: int) -> None: ...
    @typing.overload
    def negate(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def negate(self, arg0: Box, arg1: int) -> None: ...
    @typing.overload
    def negate(self, arg0: Box, arg1: int, arg2: int, arg3: int) -> None: ...
    def norm0(self, arg0: int, arg1: int, arg2: bool, arg3: bool) -> float: ...
    @typing.overload
    def norm1(self, arg0: int, arg1: Periodicity, arg2: bool) -> float: ...
    @typing.overload
    def norm1(self, arg0: int, arg1: int, arg2: bool) -> float: ...
    @typing.overload
    def norm1(self, arg0: Vector_int, arg1: int, arg2: bool) -> Vector_Real: ...
    @typing.overload
    def norm2(self, arg0: int) -> float: ...
    @typing.overload
    def norm2(self, arg0: int, arg1: Periodicity) -> float: ...
    @typing.overload
    def norm2(self, arg0: Vector_int) -> Vector_Real: ...
    def norminf(self, arg0: int, arg1: int, arg2: bool, arg3: bool) -> float: ...
    @typing.overload
    def plus(self, arg0: float, arg1: int) -> None: ...
    @typing.overload
    def plus(self, arg0: float, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def plus(self, arg0: float, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def plus(self, arg0: float, arg1: Box, arg2: int) -> None: ...
    @typing.overload
    def plus(self, arg0: float, arg1: Box, arg2: int, arg3: int, arg4: int) -> None: ...
    @typing.overload
    def plus(self, arg0: MultiFab, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def set_val(self, arg0: float) -> None: ...
    @typing.overload
    def set_val(self, arg0: float, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def set_val(self, arg0: float, arg1: int, arg2: int, arg3: int) -> None: ...
    @typing.overload
    def set_val(self, arg0: float, arg1: int, arg2: int, arg3: IntVect) -> None: ...
    def sum(self, comp: int = 0, local: bool = False) -> float:
        """
        Returns the sum of component 'comp' over the MultiFab -- no ghost cells are included.
        """
    def sum_unique(
        self, comp: int = 0, local: bool = False, period: Periodicity = ...
    ) -> float:
        """
        Same as sum with local=false, but for non-cell-centered data, thisskips non-unique points that are owned by multiple boxes.
        """
    def to_cupy(self, copy=False, order="F"):
        """

        Provide a Cupy view into a MultiFab.

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
            A list of cupy n-dimensional arrays, for each local block in the
            MultiFab.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False, order="F"):
        """

        Provide a Numpy view into a MultiFab.

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
            A list of numpy n-dimensional arrays, for each local block in the
            MultiFab.

        """
    def weighted_sync(self, arg0: MultiFab, arg1: Periodicity) -> None: ...

class PIdx:
    class IntValues:
        """
        Members:
        """

        __members__: typing.ClassVar[dict] = {}
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: int) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: int) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    class RealValues:
        """
        Members:

          w

          vx

          vy

          vz

          Ex

          Ey

          Ez
        """

        Ex: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.Ex: 4>
        Ey: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.Ey: 5>
        Ez: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.Ez: 6>
        __members__: typing.ClassVar[
            dict[str, PIdx.RealValues]
        ]  # value = {'w': <RealValues.w: 0>, 'vx': <RealValues.vx: 1>, 'vy': <RealValues.vy: 2>, 'vz': <RealValues.vz: 3>, 'Ex': <RealValues.Ex: 4>, 'Ey': <RealValues.Ey: 5>, 'Ez': <RealValues.Ez: 6>}
        vx: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.vx: 1>
        vy: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.vy: 2>
        vz: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.vz: 3>
        w: typing.ClassVar[PIdx.RealValues]  # value = <RealValues.w: 0>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: int) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: int) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

class PODVector_int_arena:
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_arena) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: int) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_int_pinned:
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_pinned) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: int) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_int_std:
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_int_std) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: int) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_arena:
    def __getitem__(self, arg0: int) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_arena) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: float) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: float) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: float) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_pinned:
    def __getitem__(self, arg0: int) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_pinned) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: float) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: float) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: float) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class PODVector_real_std:
    def __getitem__(self, arg0: int) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, size: int) -> None: ...
    @typing.overload
    def __init__(self, other: PODVector_real_std) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: int, arg1: float) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...
    def pop_back(self) -> None: ...
    def push_back(self, arg0: float) -> None: ...
    def reserve(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int) -> None: ...
    @typing.overload
    def resize(self, arg0: int, arg1: float) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def to_cupy(self, copy=False):
        """

        Provide a Cupy view into a PODVector (e.g., RealVector, IntVector).

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

        Provide a Numpy view into a PODVector (e.g., RealVector, IntVector).

        Parameters
        ----------
        self : amrex.PODVector_*
            A PODVector class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        np.array
            A 1D numpy array.

        """
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

class ParConstIterBase_0_0_4_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_4_0_arena: ...
    def __next__(self) -> ParConstIterBase_0_0_4_0_arena: ...
    def aos(self) -> ArrayOfStructs_0_0_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_arena: ...
    def soa(self) -> StructOfArrays_4_0_arena: ...
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

class ParConstIterBase_0_0_4_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_4_0_default: ...
    def __next__(self) -> ParConstIterBase_0_0_4_0_default: ...
    def aos(self) -> ArrayOfStructs_0_0_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_default: ...
    def soa(self) -> StructOfArrays_4_0_default: ...
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

class ParConstIterBase_0_0_4_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_4_0_pinned: ...
    def __next__(self) -> ParConstIterBase_0_0_4_0_pinned: ...
    def aos(self) -> ArrayOfStructs_0_0_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_pinned: ...
    def soa(self) -> StructOfArrays_4_0_pinned: ...
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

class ParConstIterBase_0_0_5_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_5_0_arena: ...
    def __next__(self) -> ParConstIterBase_0_0_5_0_arena: ...
    def aos(self) -> ArrayOfStructs_0_0_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_arena: ...
    def soa(self) -> StructOfArrays_5_0_arena: ...
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

class ParConstIterBase_0_0_5_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_5_0_default: ...
    def __next__(self) -> ParConstIterBase_0_0_5_0_default: ...
    def aos(self) -> ArrayOfStructs_0_0_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_default: ...
    def soa(self) -> StructOfArrays_5_0_default: ...
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

class ParConstIterBase_0_0_5_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_0_0_5_0_pinned: ...
    def __next__(self) -> ParConstIterBase_0_0_5_0_pinned: ...
    def aos(self) -> ArrayOfStructs_0_0_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_pinned: ...
    def soa(self) -> StructOfArrays_5_0_pinned: ...
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

class ParConstIterBase_1_1_2_1_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_1_1_2_1_arena: ...
    def __next__(self) -> ParConstIterBase_1_1_2_1_arena: ...
    def aos(self) -> ArrayOfStructs_1_1_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_arena: ...
    def soa(self) -> StructOfArrays_2_1_arena: ...
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

class ParConstIterBase_1_1_2_1_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_1_1_2_1_default: ...
    def __next__(self) -> ParConstIterBase_1_1_2_1_default: ...
    def aos(self) -> ArrayOfStructs_1_1_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_default: ...
    def soa(self) -> StructOfArrays_2_1_default: ...
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

class ParConstIterBase_1_1_2_1_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_1_1_2_1_pinned: ...
    def __next__(self) -> ParConstIterBase_1_1_2_1_pinned: ...
    def aos(self) -> ArrayOfStructs_1_1_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_pinned: ...
    def soa(self) -> StructOfArrays_2_1_pinned: ...
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

class ParConstIterBase_pureSoA_8_2_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_pureSoA_8_2_arena: ...
    def __next__(self) -> ParConstIterBase_pureSoA_8_2_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_arena: ...
    def soa(self) -> StructOfArrays_8_2_arena: ...
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

class ParConstIterBase_pureSoA_8_2_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_pureSoA_8_2_default: ...
    def __next__(self) -> ParConstIterBase_pureSoA_8_2_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_default: ...
    def soa(self) -> StructOfArrays_8_2_default: ...
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

class ParConstIterBase_pureSoA_8_2_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParConstIterBase_pureSoA_8_2_pinned: ...
    def __next__(self) -> ParConstIterBase_pureSoA_8_2_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_pinned: ...
    def soa(self) -> StructOfArrays_8_2_pinned: ...
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

class ParConstIter_0_0_4_0_arena(ParConstIterBase_0_0_4_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_0_0_4_0_default(ParConstIterBase_0_0_4_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_0_0_4_0_pinned(ParConstIterBase_0_0_4_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_0_0_5_0_arena(ParConstIterBase_0_0_5_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_0_0_5_0_default(ParConstIterBase_0_0_5_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_0_0_5_0_pinned(ParConstIterBase_0_0_5_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_1_1_2_1_arena(ParConstIterBase_1_1_2_1_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_1_1_2_1_default(ParConstIterBase_1_1_2_1_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_1_1_2_1_pinned(ParConstIterBase_1_1_2_1_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_2_arena(ParConstIterBase_pureSoA_8_2_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_2_default(ParConstIterBase_pureSoA_8_2_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParConstIter_pureSoA_8_2_pinned(ParConstIterBase_pureSoA_8_2_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIterBase_0_0_4_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_4_0_arena: ...
    def __next__(self) -> ParIterBase_0_0_4_0_arena: ...
    def aos(self) -> ArrayOfStructs_0_0_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_arena: ...
    def soa(self) -> StructOfArrays_4_0_arena: ...
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

class ParIterBase_0_0_4_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_4_0_default: ...
    def __next__(self) -> ParIterBase_0_0_4_0_default: ...
    def aos(self) -> ArrayOfStructs_0_0_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_default: ...
    def soa(self) -> StructOfArrays_4_0_default: ...
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

class ParIterBase_0_0_4_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_4_0_pinned: ...
    def __next__(self) -> ParIterBase_0_0_4_0_pinned: ...
    def aos(self) -> ArrayOfStructs_0_0_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_4_0_pinned: ...
    def soa(self) -> StructOfArrays_4_0_pinned: ...
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

class ParIterBase_0_0_5_0_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_5_0_arena: ...
    def __next__(self) -> ParIterBase_0_0_5_0_arena: ...
    def aos(self) -> ArrayOfStructs_0_0_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_arena: ...
    def soa(self) -> StructOfArrays_5_0_arena: ...
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

class ParIterBase_0_0_5_0_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_5_0_default: ...
    def __next__(self) -> ParIterBase_0_0_5_0_default: ...
    def aos(self) -> ArrayOfStructs_0_0_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_default: ...
    def soa(self) -> StructOfArrays_5_0_default: ...
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

class ParIterBase_0_0_5_0_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_0_0_5_0_pinned: ...
    def __next__(self) -> ParIterBase_0_0_5_0_pinned: ...
    def aos(self) -> ArrayOfStructs_0_0_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_5_0_pinned: ...
    def soa(self) -> StructOfArrays_5_0_pinned: ...
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

class ParIterBase_1_1_2_1_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_1_1_2_1_arena: ...
    def __next__(self) -> ParIterBase_1_1_2_1_arena: ...
    def aos(self) -> ArrayOfStructs_1_1_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_arena: ...
    def soa(self) -> StructOfArrays_2_1_arena: ...
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

class ParIterBase_1_1_2_1_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_1_1_2_1_default: ...
    def __next__(self) -> ParIterBase_1_1_2_1_default: ...
    def aos(self) -> ArrayOfStructs_1_1_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_default: ...
    def soa(self) -> StructOfArrays_2_1_default: ...
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

class ParIterBase_1_1_2_1_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_1_1_2_1_pinned: ...
    def __next__(self) -> ParIterBase_1_1_2_1_pinned: ...
    def aos(self) -> ArrayOfStructs_1_1_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_1_1_2_1_pinned: ...
    def soa(self) -> StructOfArrays_2_1_pinned: ...
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

class ParIterBase_pureSoA_8_2_arena(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_arena, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_pureSoA_8_2_arena: ...
    def __next__(self) -> ParIterBase_pureSoA_8_2_arena: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_arena: ...
    def soa(self) -> StructOfArrays_8_2_arena: ...
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

class ParIterBase_pureSoA_8_2_default(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_default, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_pureSoA_8_2_default: ...
    def __next__(self) -> ParIterBase_pureSoA_8_2_default: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_default: ...
    def soa(self) -> StructOfArrays_8_2_default: ...
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

class ParIterBase_pureSoA_8_2_pinned(MFIter):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_pinned, level: int
    ) -> None: ...
    def __iter__(self) -> ParIterBase_pureSoA_8_2_pinned: ...
    def __next__(self) -> ParIterBase_pureSoA_8_2_pinned: ...
    def geom(self, level: int) -> Geometry: ...
    def particle_tile(self) -> ParticleTile_0_0_8_2_pinned: ...
    def soa(self) -> StructOfArrays_8_2_pinned: ...
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

class ParIter_0_0_4_0_arena(ParIterBase_0_0_4_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_0_0_4_0_default(ParIterBase_0_0_4_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_0_0_4_0_pinned(ParIterBase_0_0_4_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_4_0_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_0_0_5_0_arena(ParIterBase_0_0_5_0_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_0_0_5_0_default(ParIterBase_0_0_5_0_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_0_0_5_0_pinned(ParIterBase_0_0_5_0_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_0_0_5_0_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_1_1_2_1_arena(ParIterBase_1_1_2_1_arena):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_1_1_2_1_default(ParIterBase_1_1_2_1_default):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_1_1_2_1_pinned(ParIterBase_1_1_2_1_pinned):
    is_soa_particle: typing.ClassVar[bool] = False
    def __init__(
        self, particle_container: ParticleContainer_1_1_2_1_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_2_arena(ParIterBase_pureSoA_8_2_arena):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_arena, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_2_default(ParIterBase_pureSoA_8_2_default):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_default, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParIter_pureSoA_8_2_pinned(ParIterBase_pureSoA_8_2_pinned):
    is_soa_particle: typing.ClassVar[bool] = True
    def __init__(
        self, particle_container: ParticleContainer_pureSoA_8_2_pinned, level: int
    ) -> None: ...
    def __repr__(self) -> str: ...

class ParmParse:
    @staticmethod
    def addfile(arg0: str) -> None: ...
    def __init__(self, prefix: str = "") -> None: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def add(self, arg0: str, arg1: bool) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: int) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: int) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: int) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: float) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: float) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: str) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: IntVect) -> None: ...
    @typing.overload
    def add(self, arg0: str, arg1: Box) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[int]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[int]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[int]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[float]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[float]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[str]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[IntVect]) -> None: ...
    @typing.overload
    def addarr(self, arg0: str, arg1: list[Box]) -> None: ...
    def get_bool(self, name: str, ival: int = 0) -> bool:
        """
        parses input values
        """
    def get_int(self, name: str, ival: int = 0) -> int:
        """
        parses input values
        """
    def get_real(self, name: str, ival: int = 0) -> float:
        """
        parses input values
        """
    def query_int(self, name: str, ival: int = 0) -> tuple[bool, int]:
        """
        queries input values
        """
    def remove(self, arg0: str) -> int: ...

class ParticleContainer_0_0_4_0_arena:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 4
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_4_0_arena, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_4_0_arena]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_4_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_4_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_4_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_0_0_4_0_default:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 4
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_4_0_default, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_4_0_default]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_4_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_4_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_4_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_0_0_4_0_pinned:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 4
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_4_0_pinned, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_4_0_pinned]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_4_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_4_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_4_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_0_0_5_0_arena:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 5
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_5_0_arena, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_5_0_arena]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_5_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_5_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_5_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_0_0_5_0_default:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 5
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_5_0_default, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_5_0_default]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_5_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_5_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_5_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_0_0_5_0_pinned:
    NArrayInt: typing.ClassVar[int] = 0
    NArrayReal: typing.ClassVar[int] = 5
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_5_0_pinned, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_5_0_pinned]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_0_0_5_0
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_0_0_5_0,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_0_0_5_0
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_1_1_2_1_arena:
    NArrayInt: typing.ClassVar[int] = 1
    NArrayReal: typing.ClassVar[int] = 2
    NStructInt: typing.ClassVar[int] = 1
    NStructReal: typing.ClassVar[int] = 1
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_1_1_2_1_arena, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_1_1_2_1_arena]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_1_1_2_1
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_1_1_2_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_1_1_2_1
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_1_1_2_1_default:
    NArrayInt: typing.ClassVar[int] = 1
    NArrayReal: typing.ClassVar[int] = 2
    NStructInt: typing.ClassVar[int] = 1
    NStructReal: typing.ClassVar[int] = 1
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_1_1_2_1_default, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_1_1_2_1_default]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_1_1_2_1
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_1_1_2_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_1_1_2_1
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_1_1_2_1_pinned:
    NArrayInt: typing.ClassVar[int] = 1
    NArrayReal: typing.ClassVar[int] = 2
    NStructInt: typing.ClassVar[int] = 1
    NStructReal: typing.ClassVar[int] = 1
    is_soa_particle: typing.ClassVar[bool] = False
    def AddParticlesAtLevel(
        self, particles: ParticleTile_1_1_2_1_pinned, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_1_1_2_1_pinned]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def InitOnePerCell(
        self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_1_1_2_1
    ) -> None: ...
    def InitRandom(
        self,
        arg0: int,
        arg1: int,
        arg2: ParticleInitType_1_1_2_1,
        arg3: bool,
        arg4: RealBox,
    ) -> None: ...
    def InitRandomPerBox(
        self, arg0: int, arg1: int, arg2: ParticleInitType_1_1_2_1
    ) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_pureSoA_8_2_arena:
    NArrayInt: typing.ClassVar[int] = 2
    NArrayReal: typing.ClassVar[int] = 8
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = True
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_8_2_arena, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_8_2_arena]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_pureSoA_8_2_default:
    NArrayInt: typing.ClassVar[int] = 2
    NArrayReal: typing.ClassVar[int] = 8
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = True
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_8_2_default, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_8_2_default]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleContainer_pureSoA_8_2_pinned:
    NArrayInt: typing.ClassVar[int] = 2
    NArrayReal: typing.ClassVar[int] = 8
    NStructInt: typing.ClassVar[int] = 0
    NStructReal: typing.ClassVar[int] = 0
    is_soa_particle: typing.ClassVar[bool] = True
    def AddParticlesAtLevel(
        self, particles: ParticleTile_0_0_8_2_pinned, level: int, ngrow: int = 0
    ) -> None: ...
    def ByteSpread(self) -> list[int[3]]: ...
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
    def GetParticles(
        self, level: int
    ) -> dict[tuple[int, int], ParticleTile_0_0_8_2_pinned]: ...
    def Increment(self, arg0: MultiFab, arg1: int) -> None: ...
    def NumberOfParticlesAtLevel(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
    def NumberOfParticlesInGrid(
        self, level: int, only_valid: bool = True, only_local: bool = False
    ) -> Vector_Long: ...
    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...
    def PrintCapacity(self) -> list[int[3]]: ...
    def Redistribute(
        self,
        lev_min: int = 0,
        lev_max: int = -1,
        nGrow: int = 0,
        local: int = 0,
        remove_negative: bool = True,
    ) -> None: ...
    def RemoveParticlesAtLevel(self, arg0: int) -> None: ...
    def RemoveParticlesNotAtFinestLevel(self) -> None: ...
    def ShrinkToFit(self) -> None: ...
    def SortParticlesByBin(self, arg0: IntVect) -> None: ...
    def SortParticlesByCell(self) -> None: ...
    def TotalNumberOfParticles(
        self, only_valid: bool = True, only_local: bool = False
    ) -> int: ...
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
    def clearParticles(self) -> None: ...
    def numLocalTilesAtLevel(self, arg0: int) -> int: ...
    def reserveData(self) -> None: ...
    def resizeData(self) -> None: ...
    @property
    def finest_level(self) -> int: ...

class ParticleInitType_0_0_4_0:
    is_soa_particle: typing.ClassVar[bool] = False
    int_array_data: list[int[0]]
    int_struct_data: list[int[0]]
    real_array_data: list[float[4]]
    real_struct_data: list[float[0]]
    def __init__(self) -> None: ...

class ParticleInitType_0_0_5_0:
    is_soa_particle: typing.ClassVar[bool] = False
    int_array_data: list[int[0]]
    int_struct_data: list[int[0]]
    real_array_data: list[float[5]]
    real_struct_data: list[float[0]]
    def __init__(self) -> None: ...

class ParticleInitType_1_1_2_1:
    is_soa_particle: typing.ClassVar[bool] = False
    int_array_data: list[int[1]]
    int_struct_data: list[int[1]]
    real_array_data: list[float[2]]
    real_struct_data: list[float[1]]
    def __init__(self) -> None: ...

class ParticleTileData_0_0_37_1:
    def __getitem__(self, arg0: int) -> Particle_37_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_37_1) -> None: ...
    def getSuperParticle(self, arg0: int) -> Particle_37_1: ...
    def setSuperParticle(self, arg0: Particle_37_1, arg1: int) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_0_0_4_0:
    def __getitem__(self, arg0: int) -> Particle_4_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_4_0) -> None: ...
    def getSuperParticle(self, arg0: int) -> Particle_4_0: ...
    def setSuperParticle(self, arg0: Particle_4_0, arg1: int) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_0_0_5_0:
    def __getitem__(self, arg0: int) -> Particle_5_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_5_0) -> None: ...
    def getSuperParticle(self, arg0: int) -> Particle_5_0: ...
    def setSuperParticle(self, arg0: Particle_5_0, arg1: int) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_0_0_8_2:
    def __getitem__(self, arg0: int) -> Particle_8_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_8_2) -> None: ...
    def getSuperParticle(self, arg0: int) -> Particle_8_2: ...
    def setSuperParticle(self, arg0: Particle_8_2, arg1: int) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTileData_1_1_2_1:
    def __getitem__(self, arg0: int) -> Particle_3_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_3_2) -> None: ...
    def getSuperParticle(self, arg0: int) -> Particle_3_2: ...
    def setSuperParticle(self, arg0: Particle_3_2, arg1: int) -> None: ...
    @property
    def m_num_runtime_int(self) -> int: ...
    @property
    def m_num_runtime_real(self) -> int: ...
    @property
    def m_size(self) -> int: ...

class ParticleTile_0_0_37_1_arena:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 37
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_arena: ...
    def GetStructOfArrays(self) -> StructOfArrays_37_1_arena: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_37_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_37_1) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_37_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_37_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[37]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_37_1_arena) -> None: ...

class ParticleTile_0_0_37_1_default:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 37
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_default: ...
    def GetStructOfArrays(self) -> StructOfArrays_37_1_default: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_37_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_37_1) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_37_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_37_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[37]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_37_1_default) -> None: ...

class ParticleTile_0_0_37_1_pinned:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 37
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_pinned: ...
    def GetStructOfArrays(self) -> StructOfArrays_37_1_pinned: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_37_1: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_37_1) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_37_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_37_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[37]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_37_1_pinned) -> None: ...

class ParticleTile_0_0_4_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 4
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_arena: ...
    def GetStructOfArrays(self) -> StructOfArrays_4_0_arena: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_4_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_4_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_4_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_4_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[4]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_4_0_arena) -> None: ...

class ParticleTile_0_0_4_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 4
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_default: ...
    def GetStructOfArrays(self) -> StructOfArrays_4_0_default: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_4_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_4_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_4_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_4_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[4]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_4_0_default) -> None: ...

class ParticleTile_0_0_4_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 4
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_pinned: ...
    def GetStructOfArrays(self) -> StructOfArrays_4_0_pinned: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_4_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_4_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_4_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_4_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[4]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_4_0_pinned) -> None: ...

class ParticleTile_0_0_5_0_arena:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 5
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_arena: ...
    def GetStructOfArrays(self) -> StructOfArrays_5_0_arena: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_5_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_5_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_5_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[5]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_5_0_arena) -> None: ...

class ParticleTile_0_0_5_0_default:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 5
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_default: ...
    def GetStructOfArrays(self) -> StructOfArrays_5_0_default: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_5_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_5_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_5_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[5]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_5_0_default) -> None: ...

class ParticleTile_0_0_5_0_pinned:
    NAI: typing.ClassVar[int] = 0
    NAR: typing.ClassVar[int] = 5
    def GetArrayOfStructs(self) -> ArrayOfStructs_0_0_pinned: ...
    def GetStructOfArrays(self) -> StructOfArrays_5_0_pinned: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_5_0: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_5_0) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_5_0: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_0_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_5_0) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[5]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_5_0_pinned) -> None: ...

class ParticleTile_0_0_8_2_arena:
    NAI: typing.ClassVar[int] = 2
    NAR: typing.ClassVar[int] = 8
    def GetStructOfArrays(self) -> StructOfArrays_8_2_arena: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_8_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_8_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_8_2: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def push_back(self, arg0: Particle_8_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[2]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[8]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_8_2_arena) -> None: ...

class ParticleTile_0_0_8_2_default:
    NAI: typing.ClassVar[int] = 2
    NAR: typing.ClassVar[int] = 8
    def GetStructOfArrays(self) -> StructOfArrays_8_2_default: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_8_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_8_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_8_2: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def push_back(self, arg0: Particle_8_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[2]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[8]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_8_2_default) -> None: ...

class ParticleTile_0_0_8_2_pinned:
    NAI: typing.ClassVar[int] = 2
    NAR: typing.ClassVar[int] = 8
    def GetStructOfArrays(self) -> StructOfArrays_8_2_pinned: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_8_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_8_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_0_0_8_2: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def push_back(self, arg0: Particle_8_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[2]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[8]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_0_0_8_2_pinned) -> None: ...

class ParticleTile_1_1_2_1_arena:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 2
    def GetArrayOfStructs(self) -> ArrayOfStructs_1_1_arena: ...
    def GetStructOfArrays(self) -> StructOfArrays_2_1_arena: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_3_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_3_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_1_1_2_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_1_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_3_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[2]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_1_1_2_1_arena) -> None: ...

class ParticleTile_1_1_2_1_default:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 2
    def GetArrayOfStructs(self) -> ArrayOfStructs_1_1_default: ...
    def GetStructOfArrays(self) -> StructOfArrays_2_1_default: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_3_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_3_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_1_1_2_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_1_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_3_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[2]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_1_1_2_1_default) -> None: ...

class ParticleTile_1_1_2_1_pinned:
    NAI: typing.ClassVar[int] = 1
    NAR: typing.ClassVar[int] = 2
    def GetArrayOfStructs(self) -> ArrayOfStructs_1_1_pinned: ...
    def GetStructOfArrays(self) -> StructOfArrays_2_1_pinned: ...
    def NumIntComps(self) -> int: ...
    def NumRealComps(self) -> int: ...
    def NumRuntimeIntComps(self) -> int: ...
    def NumRuntimeRealComps(self) -> int: ...
    def __getitem__(self, arg0: int) -> Particle_3_2: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: Particle_3_2) -> None: ...
    def capacity(self) -> int: ...
    def define(self, arg0: int, arg1: int) -> None: ...
    def empty(self) -> bool: ...
    def getNumNeighbors(self) -> int: ...
    def getParticleTileData(self) -> ParticleTileData_1_1_2_1: ...
    def numNeighborParticles(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    @typing.overload
    def push_back(self, arg0: Particle_1_1) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back(self, arg0: Particle_3_2) -> None:
        """
        Add one particle to this tile.
        """
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: list[float[2]]) -> None: ...
    @typing.overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float) -> None: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def shrink_to_fit(self) -> None: ...
    def size(self) -> int: ...
    def swap(self, arg0: ParticleTile_1_1_2_1_pinned) -> None: ...

class Particle_0_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 0
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> None: ...
    @typing.overload
    def get_rdata(self) -> None: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[0]]) -> None: ...

class Particle_1_1:
    NInt: typing.ClassVar[int] = 1
    NReal: typing.ClassVar[int] = 1
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> int: ...
    @typing.overload
    def get_idata(self) -> list[int[1]]: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[1]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[1]]) -> None: ...

class Particle_2_1:
    NInt: typing.ClassVar[int] = 1
    NReal: typing.ClassVar[int] = 2
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> int: ...
    @typing.overload
    def get_idata(self) -> list[int[1]]: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[2]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[2]]) -> None: ...

class Particle_37_1:
    NInt: typing.ClassVar[int] = 1
    NReal: typing.ClassVar[int] = 37
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> int: ...
    @typing.overload
    def get_idata(self) -> list[int[1]]: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[37]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[1]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[37]]) -> None: ...

class Particle_3_2:
    NInt: typing.ClassVar[int] = 2
    NReal: typing.ClassVar[int] = 3
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> int: ...
    @typing.overload
    def get_idata(self) -> list[int[2]]: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[3]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[2]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[3]]) -> None: ...

class Particle_4_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 4
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[4]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[4]]) -> None: ...

class Particle_5_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 5
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[5]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[5]]) -> None: ...

class Particle_7_0:
    NInt: typing.ClassVar[int] = 0
    NReal: typing.ClassVar[int] = 7
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> None: ...
    @typing.overload
    def get_idata(self) -> None: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[7]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[0]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[7]]) -> None: ...

class Particle_8_2:
    NInt: typing.ClassVar[int] = 2
    NReal: typing.ClassVar[int] = 8
    x: float
    y: float
    z: float
    @typing.overload
    def NextID(self) -> int: ...
    @typing.overload
    def NextID(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, *args) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, **kwargs) -> None: ...
    @typing.overload
    def __init__(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def cpu(self) -> int: ...
    @typing.overload
    def get_idata(self, arg0: int) -> int: ...
    @typing.overload
    def get_idata(self) -> list[int[2]]: ...
    @typing.overload
    def get_rdata(self, arg0: int) -> float: ...
    @typing.overload
    def get_rdata(self) -> list[float[8]]: ...
    def id(self) -> int: ...
    @typing.overload
    def pos(self, arg0: int) -> float: ...
    @typing.overload
    def pos(self) -> RealVect: ...
    @typing.overload
    def setPos(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def setPos(self, arg0: RealVect) -> None: ...
    @typing.overload
    def setPos(self, arg0: list[float[3]]) -> None: ...
    @typing.overload
    def set_idata(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def set_idata(self, arg0: IntVect) -> None: ...
    @typing.overload
    def set_idata(self, arg0: list[int[2]]) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: RealVect) -> None: ...
    @typing.overload
    def set_rdata(self, arg0: list[float[8]]) -> None: ...

class Periodicity:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def non_periodic() -> Periodicity:
        """
        Return the Periodicity object that is not periodic in any direction
        """
    def __eq__(self, arg0: Periodicity) -> bool: ...
    def __getitem__(self, dir: int) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: IntVect) -> None: ...
    def __repr__(self) -> str: ...
    def is_periodic(self, dir: int) -> bool: ...
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
    def shift_IntVect(self) -> list[IntVect]: ...

class RealBox:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        x_lo: float,
        y_lo: float,
        z_lo: float,
        x_hi: float,
        y_hi: float,
        z_hi: float,
    ) -> None: ...
    @typing.overload
    def __init__(self, a_lo: list[float[3]], a_hi: list[float[3]]) -> None: ...
    @typing.overload
    def __init__(self, bx: Box, dx: list[float[3]], base: list[float[3]]) -> None: ...
    def __repr__(self) -> str: ...
    def __str(self) -> str: ...
    @typing.overload
    def contains(self, rb: XDim3, eps: float = 0.0) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def contains(self, rb: RealVect, eps: float = 0.0) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def contains(self, rb: RealBox, eps: float = 0.0) -> bool:
        """
        Determine if RealBox contains another RealBox, within tolerance ``eps``
        """
    @typing.overload
    def contains(self, rb: list[float], eps: float = 0.0) -> bool:
        """
        Determine if RealBox contains ``pt``, within tolerance ``eps``
        """
    @typing.overload
    def hi(self, arg0: int) -> float:
        """
        Get ith component of ``xhi``
        """
    @typing.overload
    def hi(self) -> list[float[3]]:
        """
        Get all components of ``xhi``
        """
    def intersects(self, arg0: RealBox) -> bool:
        """
        determine if box intersects with a box
        """
    def length(self, arg0: int) -> float: ...
    @typing.overload
    def lo(self, arg0: int) -> float:
        """
        Get ith component of ``xlo``
        """
    @typing.overload
    def lo(self) -> list[float[3]]:
        """
        Get all components of ``xlo``
        """
    def ok(self) -> bool:
        """
        Determine if RealBox satisfies ``xlo[i]<xhi[i]`` for ``i=0,1,...,AMREX_SPACEDIM``.
        """
    @typing.overload
    def setHi(self, arg0: list[float]) -> None:
        """
        Get all components of ``xlo``
        """
    @typing.overload
    def setHi(self, arg0: int, arg1: float) -> None:
        """
        Get ith component of ``xhi``
        """
    @typing.overload
    def setLo(self, arg0: list[float]) -> None:
        """
        Get ith component of ``xlo``
        """
    @typing.overload
    def setLo(self, arg0: int, arg1: float) -> None:
        """
        Get all components of ``xlo``
        """
    def volume(self) -> float: ...
    @property
    def xhi(self) -> list[float[3]]: ...
    @property
    def xlo(self) -> list[float[3]]: ...

class RealVect:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def unit_vector() -> RealVect: ...
    @staticmethod
    def zero_vector() -> RealVect: ...
    def BASISREALV(self) -> RealVect:
        """
        return basis vector in given coordinate direction
        """
    @typing.overload
    def __add__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __add__(self, arg0: RealVect) -> RealVect: ...
    def __eq__(self, arg0: RealVect) -> bool: ...
    def __ge__(self, arg0: RealVect) -> bool: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __gt__(self, arg0: RealVect) -> bool: ...
    @typing.overload
    def __iadd__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __iadd__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __imul__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __imul__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @typing.overload
    def __init__(self, arg0: IntVect) -> None: ...
    @typing.overload
    def __init__(self, arg0: list[float]) -> None: ...
    @typing.overload
    def __init__(self, arg0: float) -> None: ...
    @typing.overload
    def __isub__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __isub__(self, arg0: RealVect) -> RealVect: ...
    def __itruediv__(self, arg0: float) -> RealVect: ...
    def __le__(self, arg0: RealVect) -> bool: ...
    def __lt__(self, arg0: RealVect) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __mul__(self, arg0: float) -> RealVect: ...
    def __ne__(self, arg0: RealVect) -> bool: ...
    def __neg__(self) -> RealVect: ...
    def __pos__(self) -> RealVect: ...
    def __radd__(self, arg0: float) -> RealVect: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg0: float) -> RealVect: ...
    def __rsub__(self, arg0: float) -> RealVect: ...
    def __rtruediv__(self, arg0: float) -> RealVect: ...
    def __setitem__(self, arg0: int, arg1: float) -> float: ...
    def __str(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: RealVect) -> RealVect: ...
    @typing.overload
    def __sub__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __truediv__(self, arg0: float) -> RealVect: ...
    @typing.overload
    def __truediv__(self, arg0: RealVect) -> RealVect: ...
    def ceil(self) -> IntVect:
        """
        Return an ``IntVect`` whose components are the std::ceil of the vector components
        """
    def crossProduct(self, arg0: RealVect) -> RealVect:
        """
        Return cross product of this vector with another
        """
    def dotProduct(self, arg0: RealVect) -> float:
        """
        Return dot product of this vector with another
        """
    def floor(self) -> IntVect:
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
    def round(self) -> IntVect:
        """
        Return an ``IntVect`` whose components are the std::round of the vector components
        """
    def scale(self, arg0: float) -> RealVect:
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

class StructOfArrays_2_1_arena:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_arena[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_arena[2]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_2_1_default:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_std[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_std[2]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_2_1_pinned:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_pinned[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_pinned[2]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_37_1_arena:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_arena[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_arena[37]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_37_1_default:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_std[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_std[37]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_37_1_pinned:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_pinned[1]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_pinned[37]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_4_0_arena:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_arena[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_arena[4]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_4_0_default:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_std[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_std[4]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_4_0_pinned:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_pinned[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_pinned[4]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_5_0_arena:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_arena[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_arena[5]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_5_0_default:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_std[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_std[5]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_5_0_pinned:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_pinned[0]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_pinned[5]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_8_2_arena:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_arena[2]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_arena[8]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_8_2_default:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_std[2]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_std[8]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

class StructOfArrays_8_2_pinned:
    @typing.overload
    def GetIntData(self) -> list[PODVector_int_pinned[2]]:
        """
        Get access to the particle Int Arrays (only compile-time components)
        """
    @typing.overload
    def GetIntData(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    @typing.overload
    def GetRealData(self) -> list[PODVector_real_pinned[8]]:
        """
        Get access to the particle Real Arrays (only compile-time components)
        """
    @typing.overload
    def GetRealData(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """
    def NumIntComps(self) -> int:
        """
        Get the number of compile-time + runtime Int components
        """
    def NumRealComps(self) -> int:
        """
        Get the number of compile-time + runtime Real components
        """
    def __init__(self) -> None: ...
    def __len__(self) -> int:
        """
        Get the number of particles
        """
    def define(self, arg0: int, arg1: int) -> None: ...
    def getNumNeighbors(self) -> int: ...
    def numParticles(self) -> int: ...
    def numRealParticles(self) -> int: ...
    def numTotalParticles(self) -> int: ...
    def resize(self, arg0: int) -> None: ...
    def setNumNeighbors(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Get the number of particles
        """
    def to_cupy(self, copy=False):
        """

        Provide Cupy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        Raises
        ------
        ImportError
            Raises an exception if cupy is not installed

        """
    def to_numpy(self, copy=False):
        """

        Provide Numpy views into a StructOfArrays.

        Parameters
        ----------
        self : amrex.StructOfArrays_*
            A StructOfArrays class in pyAMReX
        copy : bool, optional
            Copy the data if true, otherwise create a view (default).

        Returns
        -------
        namedtuple
            A tuple with real and int components that are each lists
            of 1D numpy arrays.

        """

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
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> BoxArray: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> BoxArray: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_BoxArray) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_BoxArray) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: BoxArray) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_BoxArray) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: BoxArray) -> None: ...
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
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: BoxArray) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> BoxArray:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> BoxArray:
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
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> DistributionMapping: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> DistributionMapping: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_DistributionMapping) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_DistributionMapping) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: DistributionMapping) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_DistributionMapping) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: DistributionMapping) -> None: ...
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
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: DistributionMapping) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> DistributionMapping:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> DistributionMapping:
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
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> Geometry: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> Geometry: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Geometry) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Geometry) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
    def __len__(self) -> int: ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str: ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Geometry) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Geometry) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Geometry) -> None: ...
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
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: Geometry) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> Geometry:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> Geometry:
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
    def __contains__(self, x: IntVect) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> IntVect: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_IntVect) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_IntVect) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: IntVect) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_IntVect) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: IntVect) -> None: ...
    def append(self, x: IntVect) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: IntVect) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_IntVect) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: IntVect) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> IntVect:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> IntVect:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: IntVect) -> None:
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
    def __contains__(self, x: int) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Long) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Long) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Long) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    def append(self, x: int) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: int) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_Long) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: int) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: int) -> None:
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
    def __contains__(self, x: float) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> float: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> float: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Real) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_Real) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: float) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_Real) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: float) -> None: ...
    def append(self, x: float) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: float) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_Real) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: float) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> float:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> float:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: float) -> None:
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
    def __contains__(self, x: int) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> int: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_int) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_int) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    def append(self, x: int) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: int) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector_int) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: int) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: int) -> None:
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
    def __delitem__(self, arg0: int) -> None:
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
    def __getitem__(self, arg0: int) -> str: ...
    @typing.overload
    def __getitem__(self, arg0: int) -> str: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_string) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: Vector_string) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
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
    def __setitem__(self, arg0: int, arg1: str) -> None: ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector_string) -> None:
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: str) -> None: ...
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
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: str) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> str:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> str:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: str) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def size(self) -> int: ...

class XDim3:
    x: float
    y: float
    z: float
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...

def AlmostEqual(rb1: RealBox, rb2: RealBox, eps: float = 0.0) -> bool:
    """
    Determine if two boxes are equal to within a tolerance
    """

def The_Arena() -> Arena: ...
def The_Async_Arena() -> Arena: ...
def The_Cpu_Arena() -> Arena: ...
def The_Device_Arena() -> Arena: ...
def The_Managed_Arena() -> Arena: ...
def The_Pinned_Arena() -> Arena: ...
def begin(arg0: Box) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect, arg1: IntVect) -> IntVect: ...
@typing.overload
def coarsen(arg0: Dim3, arg1: IntVect) -> Dim3: ...
@typing.overload
def coarsen(arg0: IntVect, arg1: int) -> IntVect: ...
def concatenate(root: str, num: int, mindigits: int = 5) -> str:
    """
    Builds plotfile name
    """

@typing.overload
def dtoh_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None: ...
@typing.overload
def dtoh_memcpy(
    dest: FabArray_FArrayBox,
    src: FabArray_FArrayBox,
    scomp: int,
    dcomp: int,
    ncomp: int,
) -> None: ...
def end(arg0: Box) -> Dim3: ...
@typing.overload
def finalize() -> None: ...
@typing.overload
def finalize(arg0: AMReX) -> None: ...
@typing.overload
def htod_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None: ...
@typing.overload
def htod_memcpy(
    dest: FabArray_FArrayBox,
    src: FabArray_FArrayBox,
    scomp: int,
    dcomp: int,
    ncomp: int,
) -> None: ...
def initialize(arg0: list) -> AMReX:
    """
    Initialize AMReX library
    """

def initialized() -> bool:
    """
    Returns true if there are any currently-active and initialized AMReX instances (i.e. one for which amrex::Initialize has been called, and amrex::Finalize has not). Otherwise false.
    """

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
def max(arg0: RealVect, arg1: RealVect) -> RealVect: ...
def min(arg0: RealVect, arg1: RealVect) -> RealVect: ...
def refine(arg0: Dim3, arg1: IntVect) -> Dim3: ...
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
def unpack_cpus(arg0: numpy.ndarray[numpy.uint64]) -> typing.Any: ...
def unpack_ids(arg0: numpy.ndarray[numpy.uint64]) -> typing.Any: ...
def write_single_level_plotfile(
    plotfilename: str,
    mf: MultiFab,
    varnames: Vector_string,
    geom: Geometry,
    time: float,
    level_step: int,
    versionName: str = "HyperCLaw-V1.1",
    levelPrefix: str = "Level_",
    mfPrefix: str = "Cell",
    extra_dirs: Vector_string = ...,
) -> None:
    """
    Writes single level plotfile
    """

__author__: str = "Axel Huebl, Ryan T. Sandberg, Shreyas Ananthan, David P. Grote, Revathi Jambunathan, Edoardo Zoni, Remi Lehe, Andrew Myers, Weiqun Zhang"
__license__: str = "BSD-3-Clause-LBNL"
__version__: str = "23.09"
