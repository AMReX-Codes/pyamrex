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

from collections.abc import Callable, Iterable, Iterator, Sequence
import enum
from typing import Annotated, TypeAlias, overload

import numpy
from numpy.typing import NDArray

from . import ParallelDescriptor as ParallelDescriptor


class AMReX:
    @staticmethod
    def empty() -> bool: ...

    @staticmethod
    def size() -> int: ...

    @staticmethod
    def erase(arg: AMReX, /) -> None: ...

    @staticmethod
    def top() -> AMReX: ...

class Config:
    amrex_version: str = ...
    """AMReX library version"""

    spacedim: int = ...
    """(arg: object, /) -> int"""

    verbose: int = ...
    """(arg: object, /) -> int"""

    have_eb: bool = ...
    """(arg: object, /) -> bool"""

    have_mpi: bool = ...
    """(arg: object, /) -> bool"""

    have_gpu: bool = ...
    """(arg: object, /) -> bool"""

    have_omp: bool = ...
    """(arg: object, /) -> bool"""

    have_simd: bool = ...
    """(arg: object, /) -> bool"""

    simd_size: int = ...
    """(arg: object, /) -> int"""

    gpu_backend: object = ...
    """(arg: object, /) -> object"""

    precision: str = ...
    """(arg: object, /) -> str"""

    precision_particles: str = ...
    """(arg: object, /) -> str"""

def initialize(arg: list, /) -> AMReX:
    """Initialize AMReX library"""

def initialized() -> bool:
    """
    Returns true if there are any currently-active and initialized AMReX instances (i.e. one for which amrex::Initialize has been called, and amrex::Finalize has not). Otherwise false.
    """

def size() -> int:
    """The amr stack size, the number of amr instances pushed."""

@overload
def finalize() -> None: ...

@overload
def finalize(arg: AMReX, /) -> None: ...

class Arena:
    @staticmethod
    def initialize(arg: bool, /) -> None: ...

    @staticmethod
    def print_usage(arg: bool, /) -> None: ...

    @staticmethod
    def print_usage_to_files(filename: str, message: str) -> None: ...

    @staticmethod
    def finalize() -> None: ...

    @property
    def is_device_accessible(self) -> bool: ...

    @property
    def is_host_accessible(self) -> bool: ...

    @property
    def is_managed(self) -> bool: ...

    @property
    def is_device(self) -> bool: ...

    @property
    def is_pinned(self) -> bool: ...

    def has_free_device_memory(self, sz: int) -> bool:
        """
        Does the device have enough free memory for allocating this much memory? For CPU builds, this always return true.
        """

def The_Arena() -> Arena: ...

def The_Async_Arena() -> Arena: ...

def The_Device_Arena() -> Arena: ...

def The_Managed_Arena() -> Arena: ...

def The_Pinned_Arena() -> Arena: ...

def The_Cpu_Arena() -> Arena: ...

class Dim3:
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    @property
    def x(self) -> int: ...

    @x.setter
    def x(self, arg: int, /) -> None: ...

    @property
    def y(self) -> int: ...

    @y.setter
    def y(self, arg: int, /) -> None: ...

    @property
    def z(self) -> int: ...

    @z.setter
    def z(self, arg: int, /) -> None: ...

class XDim3:
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

def almost_equal(x: float, y: float, ulp: int = 2) -> bool: ...

class IntVect1D:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: int, /) -> None: ...

    @overload
    def __init__(self, arg: Sequence[int], /) -> None: ...

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    @property
    def sum(self) -> int: ...

    @property
    def max(self) -> int: ...

    @property
    def min(self) -> int: ...

    @staticmethod
    def zero_vector() -> IntVect1D: ...

    @staticmethod
    def unit_vector() -> IntVect1D: ...

    @staticmethod
    def node_vector() -> IntVect1D: ...

    @staticmethod
    def cell_vector() -> IntVect1D: ...

    @staticmethod
    def max_vector() -> IntVect1D: ...

    @staticmethod
    def min_vector() -> IntVect1D: ...

    def dim3(self) -> Dim3: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> int: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __eq__(self, arg: int, /) -> bool: ...

    @overload
    def __eq__(self, arg: IntVect1D, /) -> bool: ...

    @overload
    def __ne__(self, arg: int, /) -> bool: ...

    @overload
    def __ne__(self, arg: IntVect1D, /) -> bool: ...

    def __lt__(self, arg: IntVect1D, /) -> bool: ...

    def __le__(self, arg: IntVect1D, /) -> bool: ...

    def __gt__(self, arg: IntVect1D, /) -> bool: ...

    def __ge__(self, arg: IntVect1D, /) -> bool: ...

    @overload
    def __add__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __add__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __sub__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __sub__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __mul__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __mul__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __truediv__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __truediv__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __iadd__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __iadd__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __isub__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __isub__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __imul__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __imul__(self, arg: IntVect1D, /) -> IntVect1D: ...

    @overload
    def __itruediv__(self, arg: int, /) -> IntVect1D: ...

    @overload
    def __itruediv__(self, arg: IntVect1D, /) -> IntVect1D: ...

    def numpy(self) -> object: ...

@overload
def coarsen(arg0: IntVect1D, arg1: IntVect1D, /) -> IntVect1D: ...

@overload
def coarsen(arg0: Dim3, arg1: IntVect1D, /) -> Dim3: ...

@overload
def coarsen(arg0: IntVect1D, arg1: int, /) -> IntVect1D: ...

@overload
def coarsen(arg0: IntVect2D, arg1: IntVect2D, /) -> IntVect2D: ...

@overload
def coarsen(arg0: Dim3, arg1: IntVect2D, /) -> Dim3: ...

@overload
def coarsen(arg0: IntVect2D, arg1: int, /) -> IntVect2D: ...

@overload
def coarsen(arg0: IntVect3D, arg1: IntVect3D, /) -> IntVect3D: ...

@overload
def coarsen(arg0: Dim3, arg1: IntVect3D, /) -> Dim3: ...

@overload
def coarsen(arg0: IntVect3D, arg1: int, /) -> IntVect3D: ...

@overload
def refine(arg0: Dim3, arg1: IntVect1D, /) -> Dim3: ...

@overload
def refine(arg0: Dim3, arg1: IntVect2D, /) -> Dim3: ...

@overload
def refine(arg0: Dim3, arg1: IntVect3D, /) -> Dim3: ...

class IntVect2D:
    @overload
    def __init__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: int, /) -> None: ...

    @overload
    def __init__(self, arg: Sequence[int], /) -> None: ...

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    @property
    def sum(self) -> int: ...

    @property
    def max(self) -> int: ...

    @property
    def min(self) -> int: ...

    @staticmethod
    def zero_vector() -> IntVect2D: ...

    @staticmethod
    def unit_vector() -> IntVect2D: ...

    @staticmethod
    def node_vector() -> IntVect2D: ...

    @staticmethod
    def cell_vector() -> IntVect2D: ...

    @staticmethod
    def max_vector() -> IntVect2D: ...

    @staticmethod
    def min_vector() -> IntVect2D: ...

    def dim3(self) -> Dim3: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> int: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __eq__(self, arg: int, /) -> bool: ...

    @overload
    def __eq__(self, arg: IntVect2D, /) -> bool: ...

    @overload
    def __ne__(self, arg: int, /) -> bool: ...

    @overload
    def __ne__(self, arg: IntVect2D, /) -> bool: ...

    def __lt__(self, arg: IntVect2D, /) -> bool: ...

    def __le__(self, arg: IntVect2D, /) -> bool: ...

    def __gt__(self, arg: IntVect2D, /) -> bool: ...

    def __ge__(self, arg: IntVect2D, /) -> bool: ...

    @overload
    def __add__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __add__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __sub__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __sub__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __mul__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __mul__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __truediv__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __truediv__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __iadd__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __iadd__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __isub__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __isub__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __imul__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __imul__(self, arg: IntVect2D, /) -> IntVect2D: ...

    @overload
    def __itruediv__(self, arg: int, /) -> IntVect2D: ...

    @overload
    def __itruediv__(self, arg: IntVect2D, /) -> IntVect2D: ...

    def numpy(self) -> object: ...

class IntVect3D:
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: int, /) -> None: ...

    @overload
    def __init__(self, arg: Sequence[int], /) -> None: ...

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    @property
    def sum(self) -> int: ...

    @property
    def max(self) -> int: ...

    @property
    def min(self) -> int: ...

    @staticmethod
    def zero_vector() -> IntVect3D: ...

    @staticmethod
    def unit_vector() -> IntVect3D: ...

    @staticmethod
    def node_vector() -> IntVect3D: ...

    @staticmethod
    def cell_vector() -> IntVect3D: ...

    @staticmethod
    def max_vector() -> IntVect3D: ...

    @staticmethod
    def min_vector() -> IntVect3D: ...

    def dim3(self) -> Dim3: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> int: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __eq__(self, arg: int, /) -> bool: ...

    @overload
    def __eq__(self, arg: IntVect3D, /) -> bool: ...

    @overload
    def __ne__(self, arg: int, /) -> bool: ...

    @overload
    def __ne__(self, arg: IntVect3D, /) -> bool: ...

    def __lt__(self, arg: IntVect3D, /) -> bool: ...

    def __le__(self, arg: IntVect3D, /) -> bool: ...

    def __gt__(self, arg: IntVect3D, /) -> bool: ...

    def __ge__(self, arg: IntVect3D, /) -> bool: ...

    @overload
    def __add__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __add__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __sub__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __sub__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __mul__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __mul__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __truediv__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __truediv__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __iadd__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __iadd__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __isub__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __isub__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __imul__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __imul__(self, arg: IntVect3D, /) -> IntVect3D: ...

    @overload
    def __itruediv__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __itruediv__(self, arg: IntVect3D, /) -> IntVect3D: ...

    def numpy(self) -> object: ...

IntVect: TypeAlias = IntVect3D

class Vector_IntVect:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_IntVect) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[IntVect3D], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_IntVect) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[IntVect3D]: ...

    @overload
    def __getitem__(self, arg: int, /) -> IntVect3D: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_IntVect: ...

    @overload
    def __getitem__(self, arg: int, /) -> IntVect3D: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: IntVect3D, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: IntVect3D, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> IntVect3D:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_IntVect, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: IntVect3D, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_IntVect, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: IntVect3D, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: IntVect3D, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: IntVect3D, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: IntVect3D, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

class IndexType:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: IndexType) -> None: ...

    @overload
    def __init__(self, arg0: IndexType.CellIndex, arg1: IndexType.CellIndex, arg2: IndexType.CellIndex, /) -> None: ...

    class CellIndex(enum.IntEnum):
        CELL = 0

        NODE = 1

    CELL: IndexType.CellIndex = CellIndex.CELL

    NODE: IndexType.CellIndex = CellIndex.NODE

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    def __getitem__(self, arg: int, /) -> int: ...

    def __len__(self) -> int: ...

    def __eq__(self, arg: IndexType, /) -> bool: ...

    def __ne__(self, arg: IndexType, /) -> bool: ...

    def __lt__(self, arg: IndexType, /) -> bool: ...

    def set(self, arg: int, /) -> None: ...

    def unset(self, arg: int, /) -> None: ...

    def test(self, arg: int, /) -> bool: ...

    def setall(self) -> None: ...

    def clear(self) -> None: ...

    def any(self) -> bool: ...

    def ok(self) -> bool: ...

    def flip(self, arg: int, /) -> None: ...

    @overload
    def cell_centered(self) -> bool: ...

    @overload
    def cell_centered(self, arg: int, /) -> bool: ...

    @overload
    def node_centered(self) -> bool: ...

    @overload
    def node_centered(self, arg: int, /) -> bool: ...

    def set_type(self, arg0: int, arg1: IndexType.CellIndex, /) -> None: ...

    @overload
    def ix_type(self) -> IntVect3D: ...

    @overload
    def ix_type(self, arg: int, /) -> IndexType.CellIndex: ...

    def to_IntVect(self) -> IntVect3D: ...

    @staticmethod
    def cell_type() -> IndexType: ...

    @staticmethod
    def node_type() -> IndexType: ...

class RealVect:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg: IntVect3D, /) -> None: ...

    @overload
    def __init__(self, arg: Sequence[float], /) -> None: ...

    @overload
    def __init__(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    def __getitem__(self, arg: int, /) -> float: ...

    def __setitem__(self, arg0: int, arg1: float, /) -> float: ...

    def __eq__(self, arg: RealVect, /) -> bool: ...

    def __ne__(self, arg: RealVect, /) -> bool: ...

    def __lt__(self, arg: RealVect, /) -> bool: ...

    def __le__(self, arg: RealVect, /) -> bool: ...

    def __gt__(self, arg: RealVect, /) -> bool: ...

    def __ge__(self, arg: RealVect, /) -> bool: ...

    @overload
    def __iadd__(self, arg: float, /) -> RealVect: ...

    @overload
    def __iadd__(self, arg: RealVect, /) -> RealVect: ...

    @overload
    def __add__(self, arg: float, /) -> RealVect: ...

    @overload
    def __add__(self, arg: RealVect, /) -> RealVect: ...

    def __radd__(self, arg: float, /) -> RealVect: ...

    @overload
    def __isub__(self, arg: float, /) -> RealVect: ...

    @overload
    def __isub__(self, arg: RealVect, /) -> RealVect: ...

    def __rsub__(self, arg: float, /) -> RealVect: ...

    @overload
    def __sub__(self, arg: RealVect, /) -> RealVect: ...

    @overload
    def __sub__(self, arg: float, /) -> RealVect: ...

    @overload
    def __imul__(self, arg: float, /) -> RealVect: ...

    @overload
    def __imul__(self, arg: RealVect, /) -> RealVect: ...

    def __rmul__(self, arg: float, /) -> RealVect: ...

    @overload
    def __mul__(self, arg: RealVect, /) -> RealVect: ...

    @overload
    def __mul__(self, arg: float, /) -> RealVect: ...

    def dotProduct(self, arg: RealVect, /) -> float:
        """Return dot product of this vector with another"""

    def crossProduct(self, arg: RealVect, /) -> RealVect:
        """Return cross product of this vector with another"""

    def __itruediv__(self, arg: float, /) -> RealVect: ...

    @overload
    def __truediv__(self, arg: float, /) -> RealVect: ...

    @overload
    def __truediv__(self, arg: RealVect, /) -> RealVect: ...

    def __rtruediv__(self, arg: float, /) -> RealVect: ...

    def scale(self, arg: float, /) -> RealVect:
        """Multiplify each component of this vector by a scalar"""

    def floor(self) -> IntVect3D:
        """
        Return an ``IntVect`` whose components are the std::floor of the vector components
        """

    def ceil(self) -> IntVect3D:
        """
        Return an ``IntVect`` whose components are the std::ceil of the vector components
        """

    def round(self) -> IntVect3D:
        """
        Return an ``IntVect`` whose components are the std::round of the vector components
        """

    def min(self, arg: RealVect, /) -> RealVect:
        """
        Replace vector with the component-wise minima of this vector and another
        """

    def max(self, arg: RealVect, /) -> RealVect:
        """
        Replace vector with the component-wise maxima of this vector and another
        """

    def __pos__(self) -> RealVect: ...

    def __neg__(self) -> RealVect: ...

    @property
    def sum(self) -> float:
        """Sum of the components of this vector"""

    @property
    def vectorLength(self) -> float:
        """Length or 2-Norm of this vector"""

    @property
    def radSquared(self) -> float:
        """Length squared of this vector"""

    @property
    def product(self) -> float:
        """Product of entries of this vector"""

    def minDir(self, arg: bool, /) -> int:
        """direction or index of minimum value of this vector"""

    def maxDir(self, arg: bool, /) -> int:
        """direction or index of maximum value of this vector"""

    @staticmethod
    def zero_vector() -> RealVect: ...

    @staticmethod
    def unit_vector() -> RealVect: ...

    def BASISREALV(self) -> RealVect:
        """return basis vector in given coordinate direction"""

def min(arg0: RealVect, arg1: RealVect, /) -> RealVect: ...

def max(arg0: RealVect, arg1: RealVect, /) -> RealVect: ...

class Direction(enum.IntEnum):
    x = 0

    y = 1

    z = 2

class Box:
    @overload
    def __init__(self, small: IntVect3D, big: IntVect3D) -> None: ...

    @overload
    def __init__(self, small: IntVect3D, big: IntVect3D, typ: IntVect3D) -> None: ...

    @overload
    def __init__(self, small: IntVect3D, big: IntVect3D, t: IndexType) -> None: ...

    @overload
    def __init__(self, small: Sequence[int], big: Sequence[int]) -> None: ...

    @overload
    def __init__(self, small: Sequence[int], big: Sequence[int], t: IndexType) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def lo_vect(self) -> IntVect3D: ...

    @lo_vect.setter
    def lo_vect(self, arg: IntVect3D, /) -> None: ...

    @property
    def hi_vect(self) -> IntVect3D: ...

    @hi_vect.setter
    def hi_vect(self, arg: IntVect3D, /) -> None: ...

    @property
    def small_end(self) -> IntVect3D: ...

    @small_end.setter
    def small_end(self, arg: IntVect3D, /) -> None: ...

    @property
    def big_end(self) -> IntVect3D: ...

    @big_end.setter
    def big_end(self, arg: IntVect3D, /) -> None: ...

    @property
    def type(self) -> IntVect3D: ...

    @type.setter
    def type(self, arg: IndexType, /) -> Box: ...

    @property
    def ix_type(self) -> IndexType: ...

    @property
    def size(self) -> IntVect3D: ...

    @overload
    def length(self) -> IntVect3D:
        """Return IntVect of lengths of the Box"""

    @overload
    def length(self, dir: int) -> int:
        """Return the length of the Box in given direction."""

    def numPts(self) -> int:
        """Return the number of points in the Box."""

    @property
    def is_empty(self) -> bool: ...

    @property
    def ok(self) -> bool: ...

    @property
    def cell_centered(self) -> bool:
        """Returns true if Box is cell-centered in all indexing directions."""

    @property
    def num_pts(self) -> int: ...

    @property
    def d_num_pts(self) -> float: ...

    @property
    def volume(self) -> int: ...

    @property
    def the_unit_box(self) -> Box: ...

    @property
    def is_square(self) -> bool: ...

    def contains(self, p: IntVect3D) -> bool:
        """Returns true if argument is contained within Box."""

    def strictly_contains(self, p: IntVect3D) -> bool:
        """Returns true if argument is strictly contained within Box."""

    def intersects(self, b: Box) -> bool:
        """
        Returns true if Boxes have non-null intersections.
        It is an error if the Boxes have different types.
        """

    def same_size(self, b: Box) -> bool:
        """
        Returns true is Boxes same size, ie translates of each other,.
        It is an error if they have different types.
        """

    def same_type(self, b: Box) -> bool:
        """Returns true if Boxes have same type."""

    def normalize(self) -> None: ...

    @overload
    def shift(self, dir: int, nzones: int) -> Box:
        """Shift this Box nzones indexing positions in coordinate direction dir."""

    @overload
    def shift(self, iv: IntVect3D) -> Box:
        """Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ..."""

    def __add__(self, arg: IntVect3D, /) -> Box: ...

    def __sub__(self, arg: IntVect3D, /) -> Box: ...

    def __iadd__(self, arg: IntVect3D, /) -> Box: ...

    def __isub__(self, arg: IntVect3D, /) -> Box: ...

    @overload
    def convert(self, typ: IndexType) -> Box:
        """
        Convert the Box from the current type into the
        argument type.  This may change the Box coordinates:
        type CELL -> NODE : increase coordinate by one on high end
        type NODE -> CELL : reduce coordinate by one on high end
        other type mappings make no change.
        """

    @overload
    def convert(self, typ: IntVect3D) -> Box: ...

    @overload
    def grow(self, n_cell: int) -> Box:
        """
        Grow Box in all directions by given amount.
        NOTE: n_cell negative shrinks the Box by that number of cells.
        """

    @overload
    def grow(self, n_cells: IntVect3D) -> Box:
        """Grow Box in each direction by specified amount."""

    @overload
    def grow(self, idir: int, n_cell: int) -> Box:
        """
        Grow the Box on the low and high end by n_cell cells
        in direction idir.
        """

    @overload
    def grow(self, d: Direction, n_cell: int) -> Box: ...

    @overload
    def grow_low(self, idir: int, n_cell: int = 1) -> Box:
        """
        Grow the Box on the low end by n_cell cells in direction idir.
        NOTE: n_cell negative shrinks the Box by that number of cells.
        """

    @overload
    def grow_low(self, d: Direction, n_cell: int = 1) -> Box: ...

    @overload
    def grow_high(self, idir: int, n_cell: int = 1) -> Box:
        """
        Grow the Box on the high end by n_cell cells in
        direction idir.  NOTE: n_cell negative shrinks the Box by that
        number of cells.
        """

    @overload
    def grow_high(self, d: Direction, n_cell: int = 1) -> Box: ...

    @overload
    def surrounding_nodes(self) -> Box:
        """Convert to NODE type in all directions."""

    @overload
    def surrounding_nodes(self, dir: int) -> Box:
        """Convert to NODE type in given direction."""

    @overload
    def surrounding_nodes(self, d: Direction) -> Box:
        """Convert to NODE type in given direction."""

    @overload
    def enclosed_cells(self) -> Box:
        """Convert to CELL type in all directions."""

    @overload
    def enclosed_cells(self, dir: int) -> Box:
        """Convert to CELL type in given direction."""

    @overload
    def enclosed_cells(self, d: Direction) -> Box:
        """Convert to CELL type in given direction."""

    def make_slab(self, direction: int, slab_index: int) -> Box:
        """Flatten the box in one direction."""

    def __iter__(self) -> Iterator[IntVect3D]: ...

    def lbound(self, arg: Box, /) -> Dim3: ...

    def ubound(self, arg: Box, /) -> Dim3: ...

    def begin(self, box: Box) -> Dim3: ...

    def end(self, box: Box) -> Dim3: ...

@overload
def lbound(arg: Box, /) -> Dim3: ...

@overload
def lbound(arg: Array4_float, /) -> Dim3: ...

@overload
def lbound(arg: Array4_double, /) -> Dim3: ...

@overload
def lbound(arg: Array4_longdouble, /) -> Dim3: ...

@overload
def lbound(arg: Array4_float_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_double_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_longdouble_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_cfloat, /) -> Dim3: ...

@overload
def lbound(arg: Array4_cdouble, /) -> Dim3: ...

@overload
def lbound(arg: Array4_cfloat_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_cdouble_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_short, /) -> Dim3: ...

@overload
def lbound(arg: Array4_int, /) -> Dim3: ...

@overload
def lbound(arg: Array4_long, /) -> Dim3: ...

@overload
def lbound(arg: Array4_longlong, /) -> Dim3: ...

@overload
def lbound(arg: Array4_short_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_int_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_long_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_longlong_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ushort, /) -> Dim3: ...

@overload
def lbound(arg: Array4_uint, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ulong, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ulonglong, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ushort_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_uint_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ulong_const, /) -> Dim3: ...

@overload
def lbound(arg: Array4_ulonglong_const, /) -> Dim3: ...

@overload
def ubound(arg: Box, /) -> Dim3: ...

@overload
def ubound(arg: Array4_float, /) -> Dim3: ...

@overload
def ubound(arg: Array4_double, /) -> Dim3: ...

@overload
def ubound(arg: Array4_longdouble, /) -> Dim3: ...

@overload
def ubound(arg: Array4_float_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_double_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_longdouble_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_cfloat, /) -> Dim3: ...

@overload
def ubound(arg: Array4_cdouble, /) -> Dim3: ...

@overload
def ubound(arg: Array4_cfloat_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_cdouble_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_short, /) -> Dim3: ...

@overload
def ubound(arg: Array4_int, /) -> Dim3: ...

@overload
def ubound(arg: Array4_long, /) -> Dim3: ...

@overload
def ubound(arg: Array4_longlong, /) -> Dim3: ...

@overload
def ubound(arg: Array4_short_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_int_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_long_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_longlong_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ushort, /) -> Dim3: ...

@overload
def ubound(arg: Array4_uint, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ulong, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ulonglong, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ushort_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_uint_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ulong_const, /) -> Dim3: ...

@overload
def ubound(arg: Array4_ulonglong_const, /) -> Dim3: ...

def begin(arg: Box, /) -> Dim3: ...

def end(arg: Box, /) -> Dim3: ...

@overload
def length(arg: Box, /) -> Dim3: ...

@overload
def length(arg: Array4_float, /) -> Dim3: ...

@overload
def length(arg: Array4_double, /) -> Dim3: ...

@overload
def length(arg: Array4_longdouble, /) -> Dim3: ...

@overload
def length(arg: Array4_float_const, /) -> Dim3: ...

@overload
def length(arg: Array4_double_const, /) -> Dim3: ...

@overload
def length(arg: Array4_longdouble_const, /) -> Dim3: ...

@overload
def length(arg: Array4_cfloat, /) -> Dim3: ...

@overload
def length(arg: Array4_cdouble, /) -> Dim3: ...

@overload
def length(arg: Array4_cfloat_const, /) -> Dim3: ...

@overload
def length(arg: Array4_cdouble_const, /) -> Dim3: ...

@overload
def length(arg: Array4_short, /) -> Dim3: ...

@overload
def length(arg: Array4_int, /) -> Dim3: ...

@overload
def length(arg: Array4_long, /) -> Dim3: ...

@overload
def length(arg: Array4_longlong, /) -> Dim3: ...

@overload
def length(arg: Array4_short_const, /) -> Dim3: ...

@overload
def length(arg: Array4_int_const, /) -> Dim3: ...

@overload
def length(arg: Array4_long_const, /) -> Dim3: ...

@overload
def length(arg: Array4_longlong_const, /) -> Dim3: ...

@overload
def length(arg: Array4_ushort, /) -> Dim3: ...

@overload
def length(arg: Array4_uint, /) -> Dim3: ...

@overload
def length(arg: Array4_ulong, /) -> Dim3: ...

@overload
def length(arg: Array4_ulonglong, /) -> Dim3: ...

@overload
def length(arg: Array4_ushort_const, /) -> Dim3: ...

@overload
def length(arg: Array4_uint_const, /) -> Dim3: ...

@overload
def length(arg: Array4_ulong_const, /) -> Dim3: ...

@overload
def length(arg: Array4_ulonglong_const, /) -> Dim3: ...

class Periodicity:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: IntVect3D, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def is_any_periodic(self) -> bool: ...

    @property
    def is_all_periodic(self) -> bool: ...

    @property
    def domain(self) -> Box:
        """Cell-centered domain Box "infinitely" long in non-periodic directions."""

    @property
    def shift_IntVect(self, arg: IntVect3D, /) -> list[IntVect3D]: ...

    def is_periodic(self, dir: int) -> bool: ...

    def __getitem__(self, dir: int) -> bool: ...

    def __eq__(self, arg: Periodicity, /) -> bool: ...

    @staticmethod
    def non_periodic() -> Periodicity:
        """Return the Periodicity object that is not periodic in any direction"""

class Array4_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_float) -> None: ...

    @overload
    def __init__(self, arg0: Array4_float, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_float, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class Array4_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_double) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class Array4_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_longdouble) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longdouble, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longdouble, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class Array4_float_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_float_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_float_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_float_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

class Array4_double_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_double_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

class Array4_longdouble_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_longdouble_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longdouble_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longdouble_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

class Array4_cfloat:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_cfloat) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cfloat, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cfloat, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.complex64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: complex, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: complex, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: complex, /) -> None: ...

class Array4_cdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_cdouble) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cdouble, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cdouble, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: complex, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: complex, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: complex, /) -> None: ...

class Array4_cfloat_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_cfloat_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cfloat_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cfloat_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.complex64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

class Array4_cdouble_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_cdouble_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cdouble_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_cdouble_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> complex: ...

class Array4_short:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_short) -> None: ...

    @overload
    def __init__(self, arg0: Array4_short, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_short, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int16], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_int:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_int) -> None: ...

    @overload
    def __init__(self, arg0: Array4_int, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_int, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int32], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_long:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_long) -> None: ...

    @overload
    def __init__(self, arg0: Array4_long, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_long, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_longlong:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_longlong) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longlong, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longlong, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_short_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_short_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_short_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_short_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int16], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_int_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_int_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_int_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_int_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int32], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_long_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_long_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_long_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_long_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_longlong_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_longlong_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longlong_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_longlong_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.int64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_ushort:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ushort) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ushort, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ushort, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint16], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_uint:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_uint) -> None: ...

    @overload
    def __init__(self, arg0: Array4_uint, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_uint, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint32], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_ulong:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ulong) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulong, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulong, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_ulonglong:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ulonglong) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulonglong, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulonglong, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint64], dict(shape=(None, None, None))], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __setitem__(self, arg0: IntVect3D, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: int, /) -> None: ...

class Array4_ushort_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ushort_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ushort_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ushort_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint16], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_uint_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_uint_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_uint_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_uint_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint32], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_ulong_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ulong_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulong_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulong_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class Array4_ulonglong_const:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Array4_ulonglong_const) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulonglong_const, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_ulonglong_const, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: Annotated[NDArray[numpy.uint64], dict(shape=(None, None, None), writable=False)], /) -> None: ...

    def __repr__(self) -> str: ...

    def index_assert(self, i: int, j: int, k: int, n: int) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def nComp(self) -> int: ...

    @property
    def num_comp(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def to_host(self) -> object: ...

    @overload
    def contains(self, i: int, j: int, k: int) -> bool: ...

    @overload
    def contains(self, iv: IntVect3D) -> bool: ...

    @overload
    def contains(self, cell: Dim3) -> bool: ...

    @overload
    def __getitem__(self, arg: IntVect3D, /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> int: ...

class SmallMatrix_6x6_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x6_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x6_F_SI1_float, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_float: ...

    def __add__(self, arg: SmallMatrix_6x6_F_SI1_float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    def __sub__(self, arg: SmallMatrix_6x6_F_SI1_float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x6_F_SI1_float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    def __neg__(self) -> SmallMatrix_6x6_F_SI1_float: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_float: ...

    def trace(self) -> float: ...

    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_float: ...

class SmallMatrix_6x1_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x1_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_float: ...

class SmallMatrix_1x6_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x6_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_float: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_float: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_float: ...

class SmallMatrix_6x6_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x6_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x6_F_SI1_double, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_double: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_double: ...

    def __add__(self, arg: SmallMatrix_6x6_F_SI1_double, /) -> SmallMatrix_6x6_F_SI1_double: ...

    def __sub__(self, arg: SmallMatrix_6x6_F_SI1_double, /) -> SmallMatrix_6x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x6_F_SI1_double, /) -> SmallMatrix_6x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    def __neg__(self) -> SmallMatrix_6x6_F_SI1_double: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_double: ...

    def trace(self) -> float: ...

    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_double: ...

class SmallMatrix_6x1_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x1_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_double: ...

class SmallMatrix_1x6_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x6_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_double: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_double: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_double: ...

class SmallMatrix_6x6_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x6_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x6_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x6_F_SI1_longdouble, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    def __add__(self, arg: SmallMatrix_6x6_F_SI1_longdouble, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    def __sub__(self, arg: SmallMatrix_6x6_F_SI1_longdouble, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x6_F_SI1_longdouble, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    def __neg__(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @staticmethod
    def identity() -> SmallMatrix_6x6_F_SI1_longdouble: ...

    def trace(self) -> float: ...

    def transpose_in_place(self) -> SmallMatrix_6x6_F_SI1_longdouble: ...

class SmallMatrix_6x1_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x1_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __add__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __sub__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __neg__(self) -> SmallMatrix_6x1_F_SI1_longdouble: ...

class SmallMatrix_1x6_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x6_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> float: ...

    @overload
    def dot(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def prod(self) -> float: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def set_val(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def sum(self) -> float: ...

    @overload
    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x1_F_SI1_longdouble: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __add__(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __sub__(self, arg: SmallMatrix_1x6_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    @overload
    def __neg__(self) -> SmallMatrix_1x6_F_SI1_longdouble: ...

class SmallMatrix_3x6_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x6_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x6_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_3x6_F_SI1_float, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_float: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x3_F_SI1_float: ...

    def __add__(self, arg: SmallMatrix_3x6_F_SI1_float, /) -> SmallMatrix_3x6_F_SI1_float: ...

    def __sub__(self, arg: SmallMatrix_3x6_F_SI1_float, /) -> SmallMatrix_3x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_float: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_float: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x3_F_SI1_float, /) -> SmallMatrix_1x6_F_SI1_float: ...

    def __neg__(self) -> SmallMatrix_3x6_F_SI1_float: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_1x3_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x3_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x3_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_1x3_F_SI1_float, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_float: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x1_F_SI1_float: ...

    def __add__(self, arg: SmallMatrix_1x3_F_SI1_float, /) -> SmallMatrix_1x3_F_SI1_float: ...

    def __sub__(self, arg: SmallMatrix_1x3_F_SI1_float, /) -> SmallMatrix_1x3_F_SI1_float: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_float: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_float: ...

    def __neg__(self) -> SmallMatrix_1x3_F_SI1_float: ...

class SmallMatrix_6x3_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x3_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x3_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x3_F_SI1_float, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_float: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x6_F_SI1_float: ...

    def __add__(self, arg: SmallMatrix_6x3_F_SI1_float, /) -> SmallMatrix_6x3_F_SI1_float: ...

    def __sub__(self, arg: SmallMatrix_6x3_F_SI1_float, /) -> SmallMatrix_6x3_F_SI1_float: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_float: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_float: ...

    def __neg__(self) -> SmallMatrix_6x3_F_SI1_float: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_3x1_F_SI1_float:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x1_F_SI1_float) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x1_F_SI1_float: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_3x1_F_SI1_float, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x3_F_SI1_float: ...

    def __add__(self, arg: SmallMatrix_3x1_F_SI1_float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    def __sub__(self, arg: SmallMatrix_3x1_F_SI1_float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_float: ...

    def __neg__(self) -> SmallMatrix_3x1_F_SI1_float: ...

class SmallMatrix_3x6_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x6_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x6_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_3x6_F_SI1_double, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_double: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x3_F_SI1_double: ...

    def __add__(self, arg: SmallMatrix_3x6_F_SI1_double, /) -> SmallMatrix_3x6_F_SI1_double: ...

    def __sub__(self, arg: SmallMatrix_3x6_F_SI1_double, /) -> SmallMatrix_3x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_double: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_double, /) -> SmallMatrix_3x1_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_double: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x3_F_SI1_double, /) -> SmallMatrix_1x6_F_SI1_double: ...

    def __neg__(self) -> SmallMatrix_3x6_F_SI1_double: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_1x3_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x3_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x3_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_1x3_F_SI1_double, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_double: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x1_F_SI1_double: ...

    def __add__(self, arg: SmallMatrix_1x3_F_SI1_double, /) -> SmallMatrix_1x3_F_SI1_double: ...

    def __sub__(self, arg: SmallMatrix_1x3_F_SI1_double, /) -> SmallMatrix_1x3_F_SI1_double: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_double: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_double: ...

    def __neg__(self) -> SmallMatrix_1x3_F_SI1_double: ...

class SmallMatrix_6x3_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x3_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x3_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x3_F_SI1_double, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_double: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x6_F_SI1_double: ...

    def __add__(self, arg: SmallMatrix_6x3_F_SI1_double, /) -> SmallMatrix_6x3_F_SI1_double: ...

    def __sub__(self, arg: SmallMatrix_6x3_F_SI1_double, /) -> SmallMatrix_6x3_F_SI1_double: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_double: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_double: ...

    def __neg__(self) -> SmallMatrix_6x3_F_SI1_double: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_3x1_F_SI1_double:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x1_F_SI1_double) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x1_F_SI1_double: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_3x1_F_SI1_double, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_double: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x3_F_SI1_double: ...

    def __add__(self, arg: SmallMatrix_3x1_F_SI1_double, /) -> SmallMatrix_3x1_F_SI1_double: ...

    def __sub__(self, arg: SmallMatrix_3x1_F_SI1_double, /) -> SmallMatrix_3x1_F_SI1_double: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_double: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_double: ...

    def __neg__(self) -> SmallMatrix_3x1_F_SI1_double: ...

class SmallMatrix_3x6_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x6_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x6_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_3x6_F_SI1_longdouble, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __add__(self, arg: SmallMatrix_3x6_F_SI1_longdouble, /) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    def __sub__(self, arg: SmallMatrix_3x6_F_SI1_longdouble, /) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    @overload
    def __mul__(self, arg: SmallMatrix_6x1_F_SI1_longdouble, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: float, /) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    @overload
    def __rmul__(self, arg: SmallMatrix_1x3_F_SI1_longdouble, /) -> SmallMatrix_1x6_F_SI1_longdouble: ...

    def __neg__(self) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_1x3_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_1x3_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_1x3_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_1x3_F_SI1_longdouble, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def __add__(self, arg: SmallMatrix_1x3_F_SI1_longdouble, /) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def __sub__(self, arg: SmallMatrix_1x3_F_SI1_longdouble, /) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def __neg__(self) -> SmallMatrix_1x3_F_SI1_longdouble: ...

class SmallMatrix_6x3_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_6x3_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_6x3_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def dot(self, arg: SmallMatrix_6x3_F_SI1_longdouble, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_3x6_F_SI1_longdouble: ...

    def __add__(self, arg: SmallMatrix_6x3_F_SI1_longdouble, /) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __sub__(self, arg: SmallMatrix_6x3_F_SI1_longdouble, /) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __neg__(self) -> SmallMatrix_6x3_F_SI1_longdouble: ...

    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

class SmallMatrix_3x1_F_SI1_longdouble:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SmallMatrix_3x1_F_SI1_longdouble) -> None: ...

    @overload
    def __init__(self, arg: object, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def row_size(self) -> int: ...

    @property
    def column_size(self) -> int: ...

    @property
    def order(self) -> str: ...

    @property
    def starting_index(self) -> int: ...

    @staticmethod
    def zero() -> SmallMatrix_3x1_F_SI1_longdouble: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> float: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: Sequence[int], arg1: float, /) -> None: ...

    def dot(self, arg: SmallMatrix_3x1_F_SI1_longdouble, /) -> float: ...

    def prod(self) -> float: ...

    def set_val(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def sum(self) -> float: ...

    @property
    def T(self) -> SmallMatrix_1x3_F_SI1_longdouble: ...

    def __add__(self, arg: SmallMatrix_3x1_F_SI1_longdouble, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def __sub__(self, arg: SmallMatrix_3x1_F_SI1_longdouble, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def __mul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def __rmul__(self, arg: float, /) -> SmallMatrix_3x1_F_SI1_longdouble: ...

    def __neg__(self) -> SmallMatrix_3x1_F_SI1_longdouble: ...

class Vector_Real:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_Real) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[float], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_Real) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_Real: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: float, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: float, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> float:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_Real, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_Real, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: float, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_int:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_int) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_int) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_int: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_int, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_Long:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_Long) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_Long) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_Long: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_Long, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_Long, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

class Vector_Box:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_Box) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[Box], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_Box) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[Box]: ...

    @overload
    def __getitem__(self, arg: int, /) -> Box: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_Box: ...

    @overload
    def __getitem__(self, arg: int, /) -> Box: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: Box, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: Box, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> Box:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_Box, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: Box, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_Box, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: Box, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: Box, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: Box, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: Box, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

class Vector_string:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_string) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[str], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_string) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    @overload
    def __getitem__(self, arg: int, /) -> str: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_string: ...

    @overload
    def __getitem__(self, arg: int, /) -> str: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: str, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: str, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> str:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_string, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_string, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: str, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

class BoxArray:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: BoxArray) -> None: ...

    @overload
    def __init__(self, arg: Box, /) -> None: ...

    @overload
    def __init__(self, arg: Vector_Box, /) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def capacity(self) -> int: ...

    @property
    def empty(self) -> bool: ...

    @property
    def numPts(self) -> int: ...

    @property
    def d_numPts(self) -> float: ...

    def define(self, arg: Box, /) -> None: ...

    def clear(self) -> None: ...

    def resize(self, arg: int, /) -> None: ...

    def cell_equal(self, arg: BoxArray, /) -> bool: ...

    @overload
    def max_size(self, arg: int, /) -> BoxArray: ...

    @overload
    def max_size(self, arg: IntVect3D, /) -> BoxArray: ...

    @overload
    def refine(self, arg: int, /) -> BoxArray: ...

    @overload
    def refine(self, arg: IntVect3D, /) -> BoxArray: ...

    @overload
    def coarsen(self, arg: IntVect3D, /) -> BoxArray: ...

    @overload
    def coarsen(self, arg: int, /) -> BoxArray: ...

    @overload
    def coarsenable(self, arg0: int, arg1: int, /) -> bool: ...

    @overload
    def coarsenable(self, arg0: IntVect3D, arg1: int, /) -> bool: ...

    @overload
    def coarsenable(self, arg0: IntVect3D, arg1: IntVect3D, /) -> bool: ...

    @overload
    def surroundingNodes(self) -> BoxArray: ...

    @overload
    def surroundingNodes(self, arg: int, /) -> BoxArray: ...

    @overload
    def enclosed_cells(self) -> BoxArray: ...

    @overload
    def enclosed_cells(self, arg: int, /) -> BoxArray: ...

    @overload
    def convert(self, arg: IndexType, /) -> BoxArray: ...

    @overload
    def convert(self, arg: IntVect3D, /) -> BoxArray: ...

    def __getitem__(self, arg: int, /) -> Box: ...

    def get(self, arg: int, /) -> Box: ...

    def minimal_box(self) -> Box: ...

    def ix_type(self) -> IndexType: ...

class Vector_BoxArray:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_BoxArray) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[BoxArray], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_BoxArray) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[BoxArray]: ...

    @overload
    def __getitem__(self, arg: int, /) -> BoxArray: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_BoxArray: ...

    @overload
    def __getitem__(self, arg: int, /) -> BoxArray: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: BoxArray, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: BoxArray, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> BoxArray:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_BoxArray, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: BoxArray, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_BoxArray, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: BoxArray, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: BoxArray, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: BoxArray, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: BoxArray, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

class ParmParse:
    def __init__(self, prefix: str = '') -> None: ...

    def __repr__(self) -> str: ...

    def remove(self, arg: str, /) -> int: ...

    @staticmethod
    def addfile(arg: str, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: bool, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: int, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: int, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: int, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: float, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: float, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: str, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: IntVect3D, /) -> None: ...

    @overload
    def add(self, arg0: str, arg1: Box, /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[int], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[int], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[int], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[float], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[float], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[str], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[IntVect3D], /) -> None: ...

    @overload
    def addarr(self, arg0: str, arg1: Sequence[Box], /) -> None: ...

    def get_bool(self, name: str, ival: int = 0) -> bool:
        """parses input values"""

    def get_int(self, name: str, ival: int = 0) -> int:
        """parses input values"""

    def get_real(self, name: str, ival: int = 0) -> float:
        """parses input values"""

    def get_str(self, name: str, ival: int = 0) -> str:
        """parses input values"""

    def query_int(self, name: str, ival: int = 0) -> tuple[bool, int]:
        """queries input values"""

    def query_str(self, name: str, ival: int = 0) -> tuple[bool, str]:
        """queries input values"""

    def pretty_print_table(self) -> None:
        """
        Write the table in a pretty way to the ostream. If there are duplicates, only the last one is printed.
        """

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

class CoordSys:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: CoordSys) -> None: ...

    class CoordType(enum.IntEnum):
        undef = -1

        cartesian = 0

        RZ = 1

        SPHERICAL = 2

    undef: CoordSys.CoordType = CoordType.undef

    cartesian: CoordSys.CoordType = CoordType.cartesian

    RZ: CoordSys.CoordType = CoordType.RZ

    SPHERICAL: CoordSys.CoordType = CoordType.SPHERICAL

    def __repr__(self) -> str: ...

    def ok(self) -> bool: ...

    def Coord(self) -> CoordSys.CoordType: ...

    def SetCoord(self, arg: CoordSys.CoordType, /) -> None: ...

    def CoordInt(self) -> int: ...

    def IsSPHERICAL(self) -> bool: ...

    def IsRZ(self) -> bool: ...

    def IsCartesian(self) -> bool: ...

class RealBox:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, x_lo: float, y_lo: float, z_lo: float, x_hi: float, y_hi: float, z_hi: float) -> None: ...

    @overload
    def __init__(self, a_lo: Sequence[float], a_hi: Sequence[float]) -> None: ...

    @overload
    def __init__(self, bx: Box, dx: Sequence[float], base: Sequence[float]) -> None: ...

    def __repr__(self) -> str: ...

    def __str(self) -> str: ...

    @property
    def xlo(self) -> list[float]: ...

    @property
    def xhi(self) -> list[float]: ...

    @overload
    def lo(self, arg: int, /) -> float:
        """Get ith component of ``xlo``"""

    @overload
    def lo(self) -> list[float]:
        """Get all components of ``xlo``"""

    @overload
    def hi(self, arg: int, /) -> float:
        """Get ith component of ``xhi``"""

    @overload
    def hi(self) -> list[float]:
        """Get all components of ``xhi``"""

    @overload
    def setLo(self, arg: Sequence[float], /) -> None:
        """Get ith component of ``xlo``"""

    @overload
    def setLo(self, arg0: int, arg1: float, /) -> None:
        """Get all components of ``xlo``"""

    @overload
    def setHi(self, arg: Sequence[float], /) -> None:
        """Get all components of ``xlo``"""

    @overload
    def setHi(self, arg0: int, arg1: float, /) -> None:
        """Get ith component of ``xhi``"""

    def length(self, arg: int, /) -> float: ...

    def ok(self) -> bool:
        """
        Determine if RealBox satisfies ``xlo[i]<xhi[i]`` for ``i=0,1,...,AMREX_SPACEDIM``.
        """

    def volume(self) -> float: ...

    @overload
    def contains(self, rb: XDim3, eps: float = 0.0) -> bool:
        """Determine if RealBox contains ``pt``, within tolerance ``eps``"""

    @overload
    def contains(self, rb: RealVect, eps: float = 0.0) -> bool:
        """Determine if RealBox contains ``pt``, within tolerance ``eps``"""

    @overload
    def contains(self, rb: RealBox, eps: float = 0.0) -> bool:
        """
        Determine if RealBox contains another RealBox, within tolerance ``eps``
        """

    @overload
    def contains(self, rb: Sequence[float], eps: float = 0.0) -> bool:
        """Determine if RealBox contains ``pt``, within tolerance ``eps``"""

    def intersects(self, arg: RealBox, /) -> bool:
        """determine if box intersects with a box"""

def AlmostEqual(rb1: RealBox, rb2: RealBox, eps: float = 0.0) -> bool:
    """Determine if two boxes are equal to within a tolerance"""

class GeometryData:
    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def prob_domain(self) -> RealBox:
        """The problem domain (real)."""

    @property
    def domain(self) -> Box:
        """The index domain."""

    @property
    def coord(self) -> int:
        """The Coordinates type."""

    @property
    def dx(self) -> list[float]:
        """The cellsize for each coordinate direction."""

    @property
    def is_periodic(self) -> list[int]:
        """Returns whether the domain is periodic in each coordinate direction."""

    @overload
    def CellSize(self) -> list[float]:
        """Returns the cellsize for each coordinate direction."""

    @overload
    def CellSize(self, arg: int, /) -> float:
        """Returns the cellsize for specified coordinate direction."""

    @overload
    def ProbLo(self) -> list[float]:
        """Returns the lo end for each coordinate direction."""

    @overload
    def ProbLo(self, arg: int, /) -> float:
        """Returns the lo end of the problem domain in specified dimension."""

    @overload
    def ProbHi(self) -> list[float]:
        """Returns the hi end for each coordinate direction."""

    @overload
    def ProbHi(self, arg: int, /) -> float:
        """Returns the hi end of the problem domain in specified dimension."""

    def Domain(self) -> Box:
        """Returns our rectangular domain"""

    @overload
    def isPeriodic(self) -> list[int]:
        """Returns whether the domain is periodic in each direction."""

    @overload
    def isPeriodic(self, arg: int, /) -> int:
        """Returns whether the domain is periodic in the given direction."""

    def Coord(self) -> int:
        """return integer coordinate type"""

class Geometry(CoordSys):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, dom: Box, rb: RealBox, coord: int, is_per: Sequence[int]) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def data(self) -> GeometryData:
        """Returns non-static copy of geometry's stored data"""

    def ResetDefaultProbDomain(self) -> None:
        """Reset default problem domain of Geometry class with a `RealBox`"""

    def ResetDefaultPeriodicity(self) -> None:
        """Reset default periodicity of Geometry class with an Array of `int`"""

    def ResetDefaultCoord(self) -> None:
        """Reset default coord of Geometry class with an Array of `int`"""

    def define(self, dom: Box, rb: RealBox, coord: int, is_per: Sequence[int]) -> None:
        """Set geometry"""

    @property
    def prob_domain(self) -> RealBox:
        """The problem domain (real)."""

    @prob_domain.setter
    def prob_domain(self, arg: RealBox, /) -> None: ...

    @overload
    def ProbLo(self, dir: int) -> float:
        """Get the lo end of the problem domain in specified direction"""

    @overload
    def ProbLo(self) -> list[float]:
        """Get the list of lo ends of the problem domain"""

    @overload
    def ProbHi(self, dir: int) -> float:
        """Get the hi end of the problem domain in specified direction"""

    @overload
    def ProbHi(self) -> list[float]:
        """Get the list of lo ends of the problem domain"""

    def ProbSize(self) -> float:
        """the overall size of the domain"""

    def ProbLength(self, arg: int, /) -> float:
        """length of problem domain in specified dimension"""

    @property
    def domain(self) -> Box:
        """The rectangular domain (index space)."""

    @domain.setter
    def domain(self, arg: Box, /) -> None: ...

    @overload
    def isPeriodic(self, arg: int, /) -> bool:
        """Is the domain periodic in the specified direction?"""

    @overload
    def isPeriodic(self) -> list[int]:
        """Return list indicating whether domain is periodic in each direction"""

    def isAnyPeriodic(self) -> bool:
        """Is domain periodic in any direction?"""

    def isAllPeriodic(self) -> bool:
        """Is domain periodic in all directions?"""

    def period(self, dir: int) -> int:
        """Return the period in the specified direction"""

    @overload
    def periodicity(self) -> Periodicity: ...

    @overload
    def periodicity(self, b: Box) -> Periodicity:
        """Return Periodicity object with lengths determined by input Box"""

    @overload
    def growNonPeriodicDomain(self, ngrow: IntVect3D) -> Box: ...

    @overload
    def growNonPeriodicDomain(self, ngrow: int) -> Box: ...

    @overload
    def growPeriodicDomain(self, ngrow: IntVect3D) -> Box: ...

    @overload
    def growPeriodicDomain(self, ngrow: int) -> Box: ...

    def setPeriodicity(self, period: Sequence[int]) -> list[int]:
        """
        Set periodicity flags and return the old flags.
        Note that, unlike Periodicity class, the flags are just boolean.
        """

    def coarsen(self, rr: IntVect3D) -> None: ...

    def refine(self, rr: IntVect3D) -> None: ...

    def outsideRoundOffDomain(self, x: float, y: float, z: float) -> bool:
        """
        Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()
        """

    def insideRoundOffDomain(self, x: float, y: float, z: float) -> bool:
        """
        Returns true if a point is inside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()
        """

class Vector_Geometry:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_Geometry) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[Geometry], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_Geometry) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[Geometry]: ...

    @overload
    def __getitem__(self, arg: int, /) -> Geometry: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_Geometry: ...

    @overload
    def __getitem__(self, arg: int, /) -> Geometry: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: Geometry, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: Geometry, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> Geometry:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_Geometry, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: Geometry, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_Geometry, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: Geometry, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def size(self) -> int: ...

class DistributionMapping:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: DistributionMapping) -> None: ...

    @overload
    def __init__(self, arg: Vector_int, /) -> None: ...

    @overload
    def __init__(self, boxes: BoxArray) -> None: ...

    @overload
    def __init__(self, boxes: BoxArray, nprocs: int) -> None: ...

    def __repr__(self) -> str: ...

    @overload
    def define(self, boxes: BoxArray) -> None: ...

    @overload
    def define(self, boxes: BoxArray, nprocs: int) -> None: ...

    @overload
    def define(self, arg: Vector_int, /) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def capacity(self) -> int: ...

    @property
    def empty(self) -> bool: ...

    @property
    def link_count(self) -> int: ...

    def ProcessorMap(self) -> Vector_int: ...

    def __getitem__(self, arg: int, /) -> int: ...

class Vector_DistributionMapping:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_DistributionMapping) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[DistributionMapping], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_DistributionMapping) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[DistributionMapping]: ...

    @overload
    def __getitem__(self, arg: int, /) -> DistributionMapping: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_DistributionMapping: ...

    @overload
    def __getitem__(self, arg: int, /) -> DistributionMapping: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: DistributionMapping, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: DistributionMapping, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> DistributionMapping:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_DistributionMapping, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: DistributionMapping, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_DistributionMapping, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: DistributionMapping, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: DistributionMapping, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: DistributionMapping, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: DistributionMapping, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

class BaseFab_Real:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Arena, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: Arena, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg: Array4_double, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double, arg1: IndexType, /) -> None: ...

    @overload
    def __init__(self, arg: Array4_double_const, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double_const, arg1: IndexType, /) -> None: ...

    def __repr__(self) -> str: ...

    def resize(self, arg0: Box, arg1: int, arg2: Arena, /) -> None: ...

    def clear(self) -> None: ...

    @overload
    def n_bytes(self) -> int: ...

    @overload
    def n_bytes(self, arg0: Box, arg1: int, /) -> int: ...

    def n_bytes_owned(self) -> int: ...

    def n_comp(self) -> int: ...

    def num_pts(self) -> int: ...

    def size(self) -> int: ...

    def box(self) -> Box: ...

    def length(self) -> IntVect3D: ...

    def small_end(self) -> IntVect3D: ...

    def big_end(self) -> IntVect3D: ...

    def lo_vect(self) -> int: ...

    def hi_vect(self) -> int: ...

    def is_allocated(self) -> bool: ...

    def array(self) -> Array4_double: ...

    def const_array(self) -> Array4_double_const: ...

    def to_host(self) -> BaseFab_Real: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

class BCType(enum.IntEnum):
    """
    Mathematical boundary condition types stored in BCRec.

    Common values are BCType.int_dir for interior cells, BCType.foextrap
    for first-order extrapolation, BCType.reflect_even and
    BCType.reflect_odd for reflective boundaries, and BCType.ext_dir or
    BCType.ext_dir_cc for external Dirichlet values supplied by the
    application.
    """

    bogus = -666

    reflect_odd = -1

    int_dir = 0

    reflect_even = 1

    foextrap = 2

    ext_dir = 3

    hoextrap = 4

    hoextrapcc = 5

    ext_dir_cc = 6

    direction_dependent = 7

    user_1 = 1001

    user_2 = 1002

    user_3 = 1003

class PhysBCType(enum.IntEnum):
    """
    Physical boundary condition categories.

    Application code maps these physical categories to mathematical
    BCType values for each field component and coordinate direction.
    """

    interior = 0

    inflow = 1

    outflow = 2

    symmetry = 3

    slipwall = 4

    noslipwall = 5

    inflowoutflow = 6

class BCRec:
    """
    Boundary condition record for one field component.

    A BCRec stores one mathematical boundary type on the low and high side
    of each coordinate direction. Pass lists of length Config.spacedim for
    lo and hi, usually using BCType enum values.
    """

    @overload
    def __init__(self) -> None:
        """
        Create a BCRec initialized to BCType.bogus on every face.

        Set all low and high entries before using this record in a fill
        operation.
        """

    @overload
    def __init__(self, lo: Sequence[int], hi: Sequence[int]) -> None:
        """
        Create a BCRec from low-side and high-side boundary types.

        Args:
            lo: Sequence of Config.spacedim BCType or integer values for the
                low side of each coordinate direction.
            hi: Sequence of Config.spacedim BCType or integer values for the
                high side of each coordinate direction.
        """

    @overload
    def __init__(self, bx: Box, domain: Box, bc_domain: BCRec) -> None:
        """
        Create the BCRec for a sub-box from a domain BCRec.

        For each face, the returned record inherits bc_domain when bx touches
        the physical domain boundary and uses BCType.int_dir otherwise.

        Args:
            bx: Box to classify.
            domain: Physical domain box.
            bc_domain: Boundary record for the full domain.
        """

    def __repr__(self) -> str: ...

    def set_lo(self, dir: int, bc_type: int) -> None:
        """
        Set the low-side boundary type in one direction.

        Args:
            dir: Coordinate direction, from 0 to Config.spacedim - 1.
            bc_type: BCType or integer boundary value.
        """

    def set_hi(self, dir: int, bc_type: int) -> None:
        """
        Set the high-side boundary type in one direction.

        Args:
            dir: Coordinate direction, from 0 to Config.spacedim - 1.
            bc_type: BCType or integer boundary value.
        """

    @overload
    def lo(self) -> list[int]:
        """Return low-side boundary types as a list."""

    @overload
    def lo(self, dir: int) -> int:
        """Return the low-side boundary type in one direction."""

    @overload
    def hi(self) -> list[int]:
        """Return high-side boundary types as a list."""

    @overload
    def hi(self, dir: int) -> int:
        """Return the high-side boundary type in one direction."""

    def vect(self) -> list[int]:
        """
        Return all boundary types as low-side entries followed by high-side entries.
        """

    def data(self) -> list[int]:
        """
        Return all boundary types as low-side entries followed by high-side entries.
        """

    def __eq__(self, arg: BCRec, /) -> bool: ...

    def __ne__(self, arg: BCRec, /) -> bool: ...

class Vector_BCRec:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Vector_BCRec) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[BCRec], /) -> None:
        """Construct from an iterable object"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Vector_BCRec) -> None: ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[BCRec]: ...

    @overload
    def __getitem__(self, arg: int, /) -> BCRec: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Vector_BCRec: ...

    @overload
    def __getitem__(self, arg: int, /) -> BCRec: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: BCRec, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: BCRec, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> BCRec:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Vector_BCRec, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: BCRec, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Vector_BCRec, /) -> None: ...

    @overload
    def __setitem__(self, arg0: int, arg1: BCRec, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: BCRec, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: BCRec, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: BCRec, /) -> None:
        """Remove first occurrence of `arg`."""

    def size(self) -> int: ...

@overload
def setBC(bx: Box, domain: Box, bc_domain: BCRec) -> BCRec:
    """
    Return the BCRec for a box from a domain BCRec.

    For each face, the returned record inherits bc_domain when bx touches
    the physical domain boundary and uses BCType.int_dir otherwise.

    Args:
        bx: Box to classify.
        domain: Physical domain box.
        bc_domain: Boundary record for the full domain.
    """

@overload
def setBC(bx: Box, domain: Box, src_comp: int, dest_comp: int, ncomp: int, bc_domain: Vector_BCRec) -> Vector_BCRec:
    """
    Return component boundary records for a box.

    The returned Vector_BCRec has size dest_comp + ncomp. Components in
    the interval [dest_comp, dest_comp + ncomp) are populated from
    bc_domain[src_comp:src_comp + ncomp]. Earlier destination entries are
    left at their default BCType.bogus values.

    Args:
        bx: Box to classify.
        domain: Physical domain box.
        src_comp: First component to read from bc_domain.
        dest_comp: First component to write in the returned Vector_BCRec.
        ncomp: Number of component records to populate.
        bc_domain: Domain boundary records for source components.
    """

class FArrayBox(BaseFab_Real):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: Arena, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: Arena, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: bool, arg3: bool, arg4: Arena, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: Box, arg1: int, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg: Array4_double, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double, arg1: IndexType, /) -> None: ...

    @overload
    def __init__(self, arg: Array4_double_const, /) -> None: ...

    @overload
    def __init__(self, arg0: Array4_double_const, arg1: IndexType, /) -> None: ...

    def __repr__(self) -> str: ...

class MFIter:
    @overload
    def __init__(self, arg: FabArrayBase, /) -> None: ...

    @overload
    def __init__(self, arg0: FabArrayBase, arg1: MFItInfo, /) -> None: ...

    @overload
    def __init__(self, arg: MultiFab, /) -> None: ...

    @overload
    def __init__(self, arg0: MultiFab, arg1: MFItInfo, /) -> None: ...

    @overload
    def __init__(self, arg: iMultiFab, /) -> None: ...

    @overload
    def __init__(self, arg0: iMultiFab, arg1: MFItInfo, /) -> None: ...

    def __repr__(self) -> str: ...

    def finalize(self) -> None: ...

    @overload
    def tilebox(self) -> Box: ...

    @overload
    def tilebox(self, arg: IntVect3D, /) -> Box: ...

    @overload
    def tilebox(self, arg0: IntVect3D, arg1: IntVect3D, /) -> Box: ...

    def validbox(self) -> Box: ...

    def fabbox(self) -> Box: ...

    def nodaltilebox(self, dir: int = -1) -> Box: ...

    def growntilebox(self, ng: IntVect3D = -1000000) -> Box: ...

    @overload
    def grownnodaltilebox(self, int: int = -1, ng: int = -1000000) -> Box: ...

    @overload
    def grownnodaltilebox(self, int: int, ng: IntVect3D) -> Box: ...

    @property
    def is_valid(self) -> bool: ...

    @property
    def index(self) -> int: ...

    @property
    def length(self) -> int: ...

class FabArrayBase:
    @property
    def is_all_cell_centered(self) -> bool: ...

    @property
    def is_all_nodal(self) -> bool: ...

    def is_nodal(self, arg: int, /) -> bool: ...

    @property
    def nComp(self) -> int:
        """
        Return number of variables (aka components) associated with each point.
        """

    @property
    def num_comp(self) -> int:
        """
        Return number of variables (aka components) associated with each point.
        """

    @property
    def size(self) -> int:
        """Return the number of FABs in the FabArray."""

    def __len__(self) -> int:
        """Return the number of FABs in the FabArray."""

    @property
    def n_grow_vect(self) -> IntVect3D:
        """
        Return the grow factor (per direction) that defines the region of definition.
        """

class FabFactory_IArrayBox:
    pass

class FabFactory_FArrayBox:
    pass

class FabArray_IArrayBox(FabArrayBase):
    def clear(self) -> None: ...

    def ok(self) -> bool: ...

    @property
    def arena(self) -> Arena:
        """Provides access to the Arena this FabArray was build with."""

    @property
    def has_EB_fab_factory(self) -> bool: ...

    @property
    def factory(self) -> FabFactory_IArrayBox: ...

    def array(self, arg: MFIter, /) -> Array4_int: ...

    def const_array(self, arg: MFIter, /) -> Array4_int_const: ...

    @overload
    def set_val(self, val: int) -> None:
        """Set all components in the entire region of each FAB to val."""

    @overload
    def set_val(self, val: int, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """

    @overload
    def set_val(self, val: int, comp: int, num_comp: int, nghost: IntVect3D) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """

    @overload
    def set_val(self, val: int, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """

    @overload
    def set_val(self, val: int, region: Box, comp: int, num_comp: int, nghost: IntVect3D) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """

    @overload
    def abs(self, comp: int, ncomp: int, nghost: int = 0) -> None: ...

    @overload
    def abs(self, comp: int, ncomp: int, nghost: IntVect3D) -> None: ...

    def saxpy(self, a: int, x: FabArray_IArrayBox, x_comp: int, comp: int, ncomp: int, nghost: IntVect3D) -> None:
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

    def xpay(self, a: int, x: FabArray_IArrayBox, xcomp: int, comp: int, ncomp: int, nghost: IntVect3D) -> None:
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

    def lin_comb(self, a: int, x: FabArray_IArrayBox, xcomp: int, b: int, y: FabArray_IArrayBox, ycomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None:
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

    def sum(self, comp: int, nghost: IntVect3D, local: bool) -> int:
        """Returns the sum of component 'comp'"""

    @overload
    def sum_boundary(self, period: Periodicity, deterministic: bool = False) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, dst_nghost: IntVect3D, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
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

    @overload
    def override_sync(self, scomp: int, ncomp: int, period: Periodicity) -> None: ...

    @overload
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

    @overload
    def fill_boundary(self, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, nghost: IntVect3D, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, period: Periodicity, cross: bool = False) -> None: ...

class FabArray_FArrayBox(FabArrayBase):
    def clear(self) -> None: ...

    def ok(self) -> bool: ...

    @property
    def arena(self) -> Arena:
        """Provides access to the Arena this FabArray was build with."""

    @property
    def has_EB_fab_factory(self) -> bool: ...

    @property
    def factory(self) -> FabFactory_FArrayBox: ...

    def array(self, arg: MFIter, /) -> Array4_double: ...

    def const_array(self, arg: MFIter, /) -> Array4_double_const: ...

    @overload
    def set_val(self, val: float) -> None:
        """Set all components in the entire region of each FAB to val."""

    @overload
    def set_val(self, val: float, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """

    @overload
    def set_val(self, val: float, comp: int, num_comp: int, nghost: IntVect3D) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp to val.
        Also set the value of nghost boundary cells.
        """

    @overload
    def set_val(self, val: float, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """

    @overload
    def set_val(self, val: float, region: Box, comp: int, num_comp: int, nghost: IntVect3D) -> None:
        """
        Set the value of num_comp components in the valid region of
        each FAB in the FabArray, starting at component comp, as well
        as nghost boundary cells, to val, provided they also intersect
        with the Box region.
        """

    @overload
    def abs(self, comp: int, ncomp: int, nghost: int = 0) -> None: ...

    @overload
    def abs(self, comp: int, ncomp: int, nghost: IntVect3D) -> None: ...

    def saxpy(self, a: float, x: FabArray_FArrayBox, x_comp: int, comp: int, ncomp: int, nghost: IntVect3D) -> None:
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

    def xpay(self, a: float, x: FabArray_FArrayBox, xcomp: int, comp: int, ncomp: int, nghost: IntVect3D) -> None:
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

    def lin_comb(self, a: float, x: FabArray_FArrayBox, xcomp: int, b: float, y: FabArray_FArrayBox, ycomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None:
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

    def sum(self, comp: int, nghost: IntVect3D, local: bool) -> float:
        """Returns the sum of component 'comp'"""

    @overload
    def sum_boundary(self, period: Periodicity, deterministic: bool = False) -> None:
        """
        Sum values in overlapped cells.  The destination is limited to valid cells.
        """

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
    def sum_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, dst_nghost: IntVect3D, period: Periodicity, deterministic: bool = False) -> None: ...

    @overload
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

    @overload
    def override_sync(self, scomp: int, ncomp: int, period: Periodicity) -> None: ...

    @overload
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

    @overload
    def fill_boundary(self, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, nghost: IntVect3D, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, period: Periodicity, cross: bool = False) -> None: ...

    @overload
    def fill_boundary(self, scomp: int, ncomp: int, nghost: IntVect3D, period: Periodicity, cross: bool = False) -> None: ...

class MFInfo:
    def __init__(self) -> None: ...

    @property
    def alloc(self) -> bool: ...

    @alloc.setter
    def alloc(self, arg: bool, /) -> None: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    @property
    def tags(self) -> Vector_string: ...

    @tags.setter
    def tags(self, arg: Vector_string, /) -> None: ...

    def set_alloc(self, arg: bool, /) -> MFInfo: ...

    def set_arena(self, arg: Arena, /) -> MFInfo: ...

    def set_tag(self, arg: str, /) -> None: ...

class MFItInfo:
    def __init__(self) -> None: ...

    @property
    def do_tiling(self) -> bool: ...

    @do_tiling.setter
    def do_tiling(self, arg: bool, /) -> None: ...

    @property
    def dynamic(self) -> bool: ...

    @dynamic.setter
    def dynamic(self, arg: bool, /) -> None: ...

    @property
    def device_sync(self) -> bool: ...

    @device_sync.setter
    def device_sync(self, arg: bool, /) -> None: ...

    @property
    def num_streams(self) -> int: ...

    @num_streams.setter
    def num_streams(self, arg: int, /) -> None: ...

    @property
    def tilesize(self) -> IntVect3D: ...

    @tilesize.setter
    def tilesize(self, arg: IntVect3D, /) -> None: ...

    def enable_tiling(self, ts: IntVect3D) -> MFItInfo: ...

    def set_dynamic(self, f: bool) -> MFItInfo: ...

    def disable_device_sync(self) -> MFItInfo: ...

    def set_device_sync(self, f: bool) -> MFItInfo: ...

    def set_num_streams(self, n: int) -> MFItInfo: ...

    def use_default_stream(self) -> MFItInfo: ...

class iMultiFab(FabArray_IArrayBox):
    @overload
    def __init__(self) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions
                    inherited from FabArray.
        """

    @overload
    def __init__(self, a: Arena) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions.
                    If ``define`` is called later with a nullptr as MFInfo's arena, the
                    default Arena ``a`` will be used.  If the arena in MFInfo is not a
                    nullptr, the MFInfo's arena will be used.
        """

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int, info: MFInfo, factory: FabFactory_IArrayBox) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int, info: MFInfo) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D, info: MFInfo) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D, info: MFInfo, factory: FabFactory_IArrayBox) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def min(self, comp: int = 0, nghost: int = 0, local: bool = False) -> int:
        """
        Returns the minimum value of the specified component of the (i)MultiFab.
        """

    @overload
    def min(self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False) -> int:
        """
        Returns the minimum value of the specified component of the (i)MultiFab over the region.
        """

    @overload
    def max(self, comp: int = 0, nghost: int = 0, local: bool = False) -> int:
        """
        Returns the maximum value of the specified component of the (i)MultiFab.
        """

    @overload
    def max(self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False) -> int:
        """
        Returns the maximum value of the specified component of the (i)MultiFab over the region.
        """

    def minIndex(self, arg0: int, arg1: int, /) -> IntVect3D: ...

    def maxIndex(self, arg0: int, arg1: int, /) -> IntVect3D: ...

    @overload
    def sum(self, comp: int = 0, local: bool = False) -> int:
        """
        Returns the sum of component 'comp' over the (i)MultiFab -- no ghost cells are included.
        """

    @overload
    def sum(self, region: Box, comp: int = 0, local: bool = False) -> int:
        """
        Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included.
        """

    @overload
    def plus(self, val: int, nghost: int = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab.  The value
        of nghost specifies the number of cells in the boundary
        region that should be modified.
        """

    @overload
    def plus(self, val: int, comp: int, num_comp: int, nghost: int = 0) -> None:
        r"""
        Adds the scalar value \p val to the value of each cell in the
        specified subregion of the MultiFab.

        The subregion consists of the \p num_comp components starting at component \p comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """

    @overload
    def plus(self, val: int, region: Box, nghost: int = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab, that also
        intersects the Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """

    @overload
    def plus(self, val: int, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of plus(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """

    @overload
    def plus(self, mf: iMultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        This function adds the values of the cells in mf to the corresponding
        cells of this MultiFab.  mf is required to have the same BoxArray or
        "valid region" as this MultiFab.  The addition is done only to num_comp
        components, starting with component number strt_comp.  The parameter
        nghost specifies the number of boundary cells that will be modified.
        If nghost == 0, only the valid region of each FArrayBox will be
        modified.
        """

    def minus(self, mf: iMultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        This function subtracts the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The subtraction is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.
        """

    def divi(self, mf: iMultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
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

    @overload
    def mult(self, val: int, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val).
        The value of nghost specifies the number of cells in the
        boundary region that should be modified.
        """

    @overload
    def mult(self, val: int, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the specified subregion of the
        MultiFab by the scalar val (a[i] <- a[i]*val). The subregion
        consists of the num_comp components starting at component comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """

    @overload
    def mult(self, val: int, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of mult(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """

    @overload
    def mult(self, val: int, region: Box, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """

    @overload
    def negate(self, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab.  The value of nghost specifies the number of
        cells in the boundary region that should be modified.
        """

    @overload
    def negate(self, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the specified subregion of
        the MultiFab.  The subregion consists of the num_comp
        components starting at component comp.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """

    @overload
    def negate(self, region: Box, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab that also intersects the Box region.  The value
        of nghost specifies the number of cells in the boundary region
        that should be modified.
        """

    @overload
    def negate(self, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of negate(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """

    @overload
    def add(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def add(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    def copymf(self, src: iMultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: int) -> None:
        """
        Copy from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray. The copy is local
        """

    @overload
    def subtract(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def subtract(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def multiply(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def multiply(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def divide(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def divide(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def swap(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """

    @overload
    def swap(self, src: iMultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    def box_array(self) -> BoxArray: ...

    def dm(self) -> DistributionMapping: ...

    @property
    def n_comp(self) -> int: ...

    @property
    def n_grow_vect(self) -> IntVect3D: ...

    @staticmethod
    def initialize() -> None: ...

    @staticmethod
    def finalize() -> None: ...

@overload
def copy_mfab(dst: iMultiFab, src: iMultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: int) -> None: ...

@overload
def copy_mfab(dst: iMultiFab, src: iMultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: IntVect3D) -> None: ...

@overload
def copy_mfab(dst: MultiFab, src: MultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: int) -> None: ...

@overload
def copy_mfab(dst: MultiFab, src: MultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: IntVect3D) -> None: ...

@overload
def htod_memcpy(dest: FabArray_IArrayBox, src: FabArray_IArrayBox) -> None:
    """Copy from a host to device FabArray."""

@overload
def htod_memcpy(dest: FabArray_IArrayBox, src: FabArray_IArrayBox, scomp: int, dcomp: int, ncomp: int) -> None:
    """
    Copy from a host to device FabArray for a specific (number of) component(s).
    """

@overload
def htod_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None:
    """Copy from a host to device FabArray."""

@overload
def htod_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox, scomp: int, dcomp: int, ncomp: int) -> None:
    """
    Copy from a host to device FabArray for a specific (number of) component(s).
    """

@overload
def dtoh_memcpy(dest: FabArray_IArrayBox, src: FabArray_IArrayBox) -> None:
    """Copy from a device to host FabArray."""

@overload
def dtoh_memcpy(dest: FabArray_IArrayBox, src: FabArray_IArrayBox, scomp: int, dcomp: int, ncomp: int) -> None:
    """
    Copy from a device to host FabArray for a specific (number of) component(s).
    """

@overload
def dtoh_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox) -> None:
    """Copy from a device to host FabArray."""

@overload
def dtoh_memcpy(dest: FabArray_FArrayBox, src: FabArray_FArrayBox, scomp: int, dcomp: int, ncomp: int) -> None:
    """
    Copy from a device to host FabArray for a specific (number of) component(s).
    """

class MultiFab(FabArray_FArrayBox):
    @overload
    def __init__(self) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions
                    inherited from FabArray.
        """

    @overload
    def __init__(self, a: Arena) -> None:
        """
        Constructs an empty (i)MultiFab.

                    Data can be defined at a later time using the define member functions.
                    If ``define`` is called later with a nullptr as MFInfo's arena, the
                    default Arena ``a`` will be used.  If the arena in MFInfo is not a
                    nullptr, the MFInfo's arena will be used.
        """

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int, info: MFInfo, factory: FabFactory_FArrayBox) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int, info: MFInfo) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: int) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D, info: MFInfo) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D, info: MFInfo, factory: FabFactory_FArrayBox) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def __init__(self, bxs: BoxArray, dm: DistributionMapping, ncomp: int, ngrow: IntVect3D) -> None:
        r"""
        Constructs an (i)MultiFab.

        The size of the FArrayBox is given by the Box grown by \p ngrow, and
        the number of components is given by \p ncomp. If \p info is set to
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

    @overload
    def min(self, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the minimum value of the specified component of the (i)MultiFab.
        """

    @overload
    def min(self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the minimum value of the specified component of the (i)MultiFab over the region.
        """

    @overload
    def max(self, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the maximum value of the specified component of the (i)MultiFab.
        """

    @overload
    def max(self, region: Box, comp: int = 0, nghost: int = 0, local: bool = False) -> float:
        """
        Returns the maximum value of the specified component of the (i)MultiFab over the region.
        """

    def minIndex(self, arg0: int, arg1: int, /) -> IntVect3D: ...

    def maxIndex(self, arg0: int, arg1: int, /) -> IntVect3D: ...

    @overload
    def norm0(self, arg0: int, arg1: int, arg2: bool, arg3: bool, /) -> float: ...

    @overload
    def norm0(self, arg0: iMultiFab, arg1: int, arg2: int, arg3: bool, /) -> float: ...

    def norminf(self, arg0: int, arg1: int, arg2: bool, arg3: bool, /) -> float: ...

    @overload
    def norm1(self, arg0: int, arg1: Periodicity, arg2: bool, /) -> float: ...

    @overload
    def norm1(self, arg0: int, arg1: int, arg2: bool, /) -> float: ...

    @overload
    def norm1(self, arg0: Vector_int, arg1: int, arg2: bool, /) -> Vector_Real: ...

    @overload
    def norm2(self, arg: int, /) -> float: ...

    @overload
    def norm2(self, arg0: int, arg1: Periodicity, /) -> float: ...

    @overload
    def norm2(self, arg: Vector_int, /) -> Vector_Real: ...

    @overload
    def sum(self, comp: int = 0, local: bool = False) -> float:
        """
        Returns the sum of component 'comp' over the (i)MultiFab -- no ghost cells are included.
        """

    @overload
    def sum(self, region: Box, comp: int = 0, local: bool = False) -> float:
        """
        Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included.
        """

    @overload
    def sum_unique(self, comp: int = 0, local: bool = False, period: Periodicity = ...) -> float:
        """
        Same as sum with local=false, but for non-cell-centered data, thisskips non-unique points that are owned by multiple boxes.
        """

    @overload
    def sum_unique(self, region: Box, comp: int = 0, local: bool = False) -> float:
        """
        Returns the unique sum of component `comp` in the given region. Non-unique points owned by multiple boxes in the MultiFab areonly added once. No ghost cells are included. This function does not takeperiodicity into account in the determination of uniqueness of points.
        """

    @overload
    def plus(self, val: float, nghost: int = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab.  The value
        of nghost specifies the number of cells in the boundary
        region that should be modified.
        """

    @overload
    def plus(self, val: float, comp: int, num_comp: int, nghost: int = 0) -> None:
        r"""
        Adds the scalar value \p val to the value of each cell in the
        specified subregion of the MultiFab.

        The subregion consists of the \p num_comp components starting at component \p comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """

    @overload
    def plus(self, val: float, region: Box, nghost: int = 0) -> None:
        """
        Adds the scalar value val to the value of each cell in the
        valid region of each component of the MultiFab, that also
        intersects the Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """

    @overload
    def plus(self, val: float, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of plus(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """

    @overload
    def plus(self, mf: MultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        This function adds the values of the cells in mf to the corresponding
        cells of this MultiFab.  mf is required to have the same BoxArray or
        "valid region" as this MultiFab.  The addition is done only to num_comp
        components, starting with component number strt_comp.  The parameter
        nghost specifies the number of boundary cells that will be modified.
        If nghost == 0, only the valid region of each FArrayBox will be
        modified.
        """

    def minus(self, mf: MultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        This function subtracts the values of the cells in mf from the
        corresponding cells of this MultiFab.  mf is required to have the
        same BoxArray or "valid region" as this MultiFab.  The subtraction is
        done only to num_comp components, starting with component number
        strt_comp.  The parameter nghost specifies the number of boundary
        cells that will be modified.  If nghost == 0, only the valid region of
        each FArrayBox will be modified.
        """

    def divi(self, mf: MultiFab, strt_comp: int, num_comp: int, nghost: int = 0) -> None:
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

    @overload
    def mult(self, val: float, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val).
        The value of nghost specifies the number of cells in the
        boundary region that should be modified.
        """

    @overload
    def mult(self, val: float, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the specified subregion of the
        MultiFab by the scalar val (a[i] <- a[i]*val). The subregion
        consists of the num_comp components starting at component comp.
        The value of nghost specifies the number of cells in the
        boundary region of each FArrayBox in the subregion that should
        be modified.
        """

    @overload
    def mult(self, val: float, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of mult(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in
        the subregion that should be modified.
        """

    @overload
    def mult(self, val: float, region: Box, nghost: int = 0) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """

    @overload
    def invert(self, numerator: float, nghost: int) -> None:
        """
        Replaces the value of each cell in the specified subregion of
        the MultiFab with its reciprocal multiplied by the value of
        numerator.  The value of nghost specifies the number of cells
        in the boundary region that should be modified.
        """

    @overload
    def invert(self, numerator: float, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Replaces the value of each cell in the specified subregion of
        the MultiFab with its reciprocal multiplied by the value of
        numerator. The subregion consists of the num_comp components
        starting at component comp.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in the
        subregion that should be modified.
        """

    @overload
    def invert(self, numerator: float, region: Box, nghost: int) -> None:
        """
        Scales the value of each cell in the valid region of each
        component of the MultiFab by the scalar val (a[i] <- a[i]*val),
        that also intersects the Box region.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """

    @overload
    def invert(self, numerator: float, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of invert(), with the
        restriction that the subregion is further constrained to the
        intersection with Box region.  The value of nghost specifies the
        number of cells in the boundary region of each FArrayBox in the
        subregion that should be modified.
        """

    @overload
    def negate(self, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab.  The value of nghost specifies the number of
        cells in the boundary region that should be modified.
        """

    @overload
    def negate(self, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the specified subregion of
        the MultiFab.  The subregion consists of the num_comp
        components starting at component comp.  The value of nghost
        specifies the number of cells in the boundary region of each
        FArrayBox in the subregion that should be modified.
        """

    @overload
    def negate(self, region: Box, nghost: int = 0) -> None:
        """
        Negates the value of each cell in the valid region of
        the MultiFab that also intersects the Box region.  The value
        of nghost specifies the number of cells in the boundary region
        that should be modified.
        """

    @overload
    def negate(self, region: Box, comp: int, num_comp: int, nghost: int = 0) -> None:
        """
        Identical to the previous version of negate(), with the
        restriction that the subregion is further constrained to
        the intersection with Box region.
        """

    @overload
    def dot(self, comp: int, y: MultiFab, y_comp: int, numcomp: int, nghost: int, local: bool = False) -> float:
        """Returns the dot product of self with another MultiFab."""

    @overload
    def dot(self, comp: int, numcomp: int, nghost: int, local: bool = False) -> float:
        """Returns the dot product with itself."""

    @overload
    def dot(self, mask: iMultiFab, comp: int, y: MultiFab, y_comp: int, numcomp: int, nghost: int, local: bool = False) -> float:
        """
        Returns the dot product of self with another MultiFab where the mask is valid.
        """

    @overload
    def add(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Add src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def add(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def copymf(self, src: MultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: int) -> None:
        """
        Copy from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray. The copy is local
        """

    @overload
    def copymf(self, src: MultiFab, srccomp: int, dstcomp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def subtract(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Subtract src from self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def subtract(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def multiply(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Multiply self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def multiply(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def divide(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Divide self by src including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        """

    @overload
    def divide(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def swap(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """
        Swap from src to self including nghost ghost cells.
        The two MultiFabs MUST have the same underlying BoxArray.
        The swap is local.
        """

    @overload
    def swap(self, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    def saxpy(self, a: float, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """self += a * src"""

    def xpay(self, a: float, src: MultiFab, srccomp: int, comp: int, numcomp: int, nghost: int) -> None:
        """self = src + a * self"""

    def lin_comb(self, a: float, x: MultiFab, x_comp: int, b: float, y: MultiFab, y_comp: int, comp: int, numcomp: int, nghost: int) -> None:
        """self = a * x + b * y"""

    @overload
    def add_product(self, src1: MultiFab, comp1: int, src2: MultiFab, comp2: int, comp: int, numcomp: int, nghost: int) -> None:
        """self += src1 * src2"""

    @overload
    def add_product(self, src1: MultiFab, comp1: int, src2: MultiFab, comp2: int, comp: int, numcomp: int, nghost: IntVect3D) -> None: ...

    @overload
    def contains_nan(self, local: bool = False) -> bool: ...

    @overload
    def contains_nan(self, scomp: int, ncomp: int, ngrow: int = 0, local: bool = False) -> bool: ...

    @overload
    def contains_nan(self, scomp: int, ncomp: int, ngrow: IntVect3D, local: bool = False) -> bool: ...

    @overload
    def contains_inf(self, local: bool = False) -> bool: ...

    @overload
    def contains_inf(self, scomp: int, ncomp: int, ngrow: int = 0, local: bool = False) -> bool: ...

    @overload
    def contains_inf(self, scomp: int, ncomp: int, ngrow: IntVect3D, local: bool = False) -> bool: ...

    def box_array(self) -> BoxArray: ...

    def dm(self) -> DistributionMapping: ...

    @property
    def n_comp(self) -> int: ...

    @property
    def n_grow_vect(self) -> IntVect3D: ...

    def average_sync(self, arg: Periodicity, /) -> None: ...

    def weighted_sync(self, arg0: MultiFab, arg1: Periodicity, /) -> None: ...

    def override_sync(self, arg0: iMultiFab, arg1: Periodicity, /) -> None: ...

    @staticmethod
    def initialize() -> None: ...

    @staticmethod
    def finalize() -> None: ...

def fill_domain_boundary(phi: MultiFab, geom: Geometry, bc: Vector_BCRec) -> None:
    """
    Fill cell-centered physical-domain ghost cells.

    This fills non-periodic ghost cells outside the physical domain for
    BCType.foextrap, BCType.hoextrap, BCType.hoextrapcc,
    BCType.reflect_even, and BCType.reflect_odd. It intentionally leaves
    BCType.ext_dir and BCType.ext_dir_cc unchanged; fill those values from
    application code, for example with PhysBCFunctUser.

    Args:
        phi: MultiFab to modify in place. All components are processed.
        geom: Geometry defining the physical domain and periodic directions.
        bc: Vector_BCRec with one record per component in phi.

    Notes:
        This function fills physical-domain ghost cells only. For multi-box
        MultiFabs, call phi.fill_boundary() separately when interior or
        periodic ghost cells also need to be valid.
    """

class PhysBCFunctNoOp:
    """
    Physical boundary condition functor that does nothing.

    Use this with FillPatch-style calls when physical-domain ghost cells do
    not need additional work, for example in fully periodic domains or when
    the caller has already filled them.
    """

    def __init__(self) -> None:
        """Create a no-op physical boundary functor."""

    def __call__(self, mf: MultiFab, dcomp: int, ncomp: int, nghost: IntVect3D, time: float, bccomp: int) -> None:
        """
        Apply the no-op boundary fill.

        The arguments match the PhysBCFunct call interface and are accepted for
        interchangeability with other physical boundary functors.

        Args:
            mf: MultiFab passed by reference.
            dcomp: First destination component.
            ncomp: Number of destination components.
            nghost: Number of ghost cells to consider in each direction.
            time: Simulation time associated with the fill.
            bccomp: First boundary-condition component.
        """

class CpuBndryFuncFab:
    """
    Host boundary-fill helper for PhysBCFunct_CpuBndryFuncFab.

    The default-constructed helper fills extrapolation and reflection
    boundaries handled by AMReX, including BCType.foextrap,
    BCType.hoextrap, BCType.hoextrapcc, BCType.reflect_even, and
    BCType.reflect_odd. It leaves BCType.ext_dir and BCType.ext_dir_cc
    unchanged; fill external Dirichlet values separately, for example with
    PhysBCFunctUser.
    """

    def __init__(self) -> None:
        """Create the default host boundary-fill helper."""

class PhysBCFunct_CpuBndryFuncFab:
    """
    Physical boundary condition functor using CpuBndryFuncFab.

    This wraps amrex::PhysBCFunct<CpuBndryFuncFab>. It applies the
    boundary types stored in a Vector_BCRec over the physical-domain ghost
    cells selected by a Geometry.
    """

    @overload
    def __init__(self) -> None:
        """
        Create an undefined physical boundary functor.

        Call define() before invoking this object.
        """

    @overload
    def __init__(self, geom: Geometry, bc: Vector_BCRec, bndry_func: CpuBndryFuncFab) -> None:
        """
        Create a physical boundary functor.

        Args:
            geom: Geometry defining the physical domain and periodic directions.
            bc: Vector_BCRec with one record per component.
            bndry_func: Boundary-fill helper, usually CpuBndryFuncFab().
        """

    def define(self, geom: Geometry, bc: Vector_BCRec, bndry_func: CpuBndryFuncFab) -> None:
        """
        Reset the geometry, component BC records, and boundary helper.

        Args:
            geom: Geometry defining the physical domain and periodic directions.
            bc: Vector_BCRec with one record per component.
            bndry_func: Boundary-fill helper, usually CpuBndryFuncFab().
        """

    def __call__(self, mf: MultiFab, dcomp: int, ncomp: int, nghost: IntVect3D, time: float, bccomp: int) -> None:
        """
        Fill physical-domain ghost cells for a component range.

        Args:
            mf: MultiFab to modify in place.
            dcomp: First destination component in mf.
            ncomp: Number of components to fill.
            nghost: Number of ghost cells to consider in each direction.
            time: Simulation time associated with the fill.
            bccomp: First component in the stored Vector_BCRec that corresponds
                to dcomp.
        """

class PhysBCFunctUser:
    """
    Physical boundary condition functor implemented in Python.

    The callback receives (mf, dcomp, ncomp, nghost, time, bccomp). It
    should fill the ghost cells of mf that lie outside the physical domain
    for the requested component range. This is the intended hook for
    application-supplied external Dirichlet values such as BCType.ext_dir
    and BCType.ext_dir_cc.

    The callback runs on the host after pending AMReX GPU stream work is
    synchronized. When called from C++, the wrapper acquires the Python GIL
    before invoking the callback.
    """

    @overload
    def __init__(self) -> None:
        """
        Create a user boundary functor with no callback.

        Calling an empty PhysBCFunctUser is a no-op.
        """

    @overload
    def __init__(self, callback: Callable) -> None:
        """
        Create a user boundary functor from a Python callback.

        Args:
            callback: Callable with signature
                callback(mf, dcomp, ncomp, nghost, time, bccomp).
        """

    def __call__(self, mf: MultiFab, dcomp: int, ncomp: int, nghost: IntVect3D, time: float, bccomp: int) -> None:
        """
        Invoke the Python physical-boundary callback.

        Args:
            mf: MultiFab to modify in place.
            dcomp: First destination component in mf.
            ncomp: Number of components the callback should fill.
            nghost: Number of ghost cells to consider in each direction.
            time: Simulation time associated with the fill.
            bccomp: First boundary-condition component corresponding to dcomp.
        """

class GrowthStrategy(enum.Enum):
    Poisson = 0

    Exact = 1

    Geometric = 2

Poisson: GrowthStrategy = GrowthStrategy.Poisson

Exact: GrowthStrategy = GrowthStrategy.Exact

Geometric: GrowthStrategy = GrowthStrategy.Geometric

class PODVector_real_pinned:
    """
    A plain-old-data (POD) vector of 'real' elements with 'pinned' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_real_pinned) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: float) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: float, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: float, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    def __getitem__(self, arg: int, /) -> float: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_real_pinned:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_real_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_real_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_real_arena:
    """
    A plain-old-data (POD) vector of 'real' elements with 'arena' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_real_arena) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: float) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: float, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: float, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    def __getitem__(self, arg: int, /) -> float: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_real_arena:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_real_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_real_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_real_std:
    """
    A plain-old-data (POD) vector of 'real' elements with 'std' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_real_std) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: float) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: float, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: float, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    def __getitem__(self, arg: int, /) -> float: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_real_std:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_real_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_real_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_real_polymorphic:
    """
    A plain-old-data (POD) vector of 'real' elements with 'polymorphic' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_real_polymorphic) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: float) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: float, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: float, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    def __getitem__(self, arg: int, /) -> float: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_real_polymorphic:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_real_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_real_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

DeviceVector_real: TypeAlias = PODVector_real_std

NonManagedDeviceVector_real: TypeAlias = PODVector_real_std

ManagedVector_real: TypeAlias = PODVector_real_std

ManagedDeviceVector_real: TypeAlias = PODVector_real_std

PinnedVector_real: TypeAlias = PODVector_real_std

AsyncVector_real: TypeAlias = PODVector_real_std

HostVector_real: TypeAlias = PODVector_real_std

PODVector_real_default: TypeAlias = PODVector_real_std

class PODVector_int_pinned:
    """
    A plain-old-data (POD) vector of 'int' elements with 'pinned' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_int_pinned) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_int_pinned:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_int_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_int_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_int_arena:
    """
    A plain-old-data (POD) vector of 'int' elements with 'arena' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_int_arena) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_int_arena:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_int_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_int_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_int_std:
    """A plain-old-data (POD) vector of 'int' elements with 'std' allocation."""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_int_std) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_int_std:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_int_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_int_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_int_polymorphic:
    """
    A plain-old-data (POD) vector of 'int' elements with 'polymorphic' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_int_polymorphic) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_int_polymorphic:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_int_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_int_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

DeviceVector_int: TypeAlias = PODVector_int_std

NonManagedDeviceVector_int: TypeAlias = PODVector_int_std

ManagedVector_int: TypeAlias = PODVector_int_std

ManagedDeviceVector_int: TypeAlias = PODVector_int_std

PinnedVector_int: TypeAlias = PODVector_int_std

AsyncVector_int: TypeAlias = PODVector_int_std

HostVector_int: TypeAlias = PODVector_int_std

PODVector_int_default: TypeAlias = PODVector_int_std

class PODVector_uint64_pinned:
    """
    A plain-old-data (POD) vector of 'uint64' elements with 'pinned' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_uint64_pinned) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_uint64_pinned:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_uint64_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_uint64_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_uint64_arena:
    """
    A plain-old-data (POD) vector of 'uint64' elements with 'arena' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_uint64_arena) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_uint64_arena:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_uint64_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_uint64_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_uint64_std:
    """
    A plain-old-data (POD) vector of 'uint64' elements with 'std' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_uint64_std) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_uint64_std:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_uint64_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_uint64_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

class PODVector_uint64_polymorphic:
    """
    A plain-old-data (POD) vector of 'uint64' elements with 'polymorphic' allocation.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, size: int) -> None: ...

    @overload
    def __init__(self, other: PODVector_uint64_polymorphic) -> None: ...

    def __repr__(self) -> str: ...

    def assign(self, value: int) -> None:
        """assign the same value to every element"""

    def push_back(self, arg: int, /) -> None: ...

    def pop_back(self) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def capacity(self) -> int: ...

    def empty(self) -> bool: ...

    @overload
    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def resize(self, new_size: int, value: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def reserve(self, capacity: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def shrink_to_fit(self) -> None: ...

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    def __getitem__(self, arg: int, /) -> int: ...

    @staticmethod
    def from_numpy(arr: object) -> PODVector_uint64_polymorphic:
        """
        Create a new PODVector from a NumPy array (or array-like).

        Always copies the data into a newly allocated PODVector. The input is cast to
        the vector's element type and made contiguous as needed. The copy into
        device-only memory uses an AMReX host-to-device copy and does not require CuPy.

        Parameters
        ----------
        arr : array_like
            Input data, convertible to a NumPy array.

        Returns
        -------
        PODVector
            A new PODVector with a copy of the data.
        """

    def to_host(self) -> PODVector_uint64_pinned:
        """
        Copy this vector into a new pinned (host) PODVector. Mirrors to_device().
        """

    def to_device(self) -> PODVector_uint64_std:
        """
        Copy this vector into a new amrex Gpu::DeviceVector (the arena allocator on GPU, std on CPU), transferring across memory spaces as needed. Mirrors to_host().
        """

DeviceVector_uint64: TypeAlias = PODVector_uint64_std

NonManagedDeviceVector_uint64: TypeAlias = PODVector_uint64_std

ManagedVector_uint64: TypeAlias = PODVector_uint64_std

ManagedDeviceVector_uint64: TypeAlias = PODVector_uint64_std

PinnedVector_uint64: TypeAlias = PODVector_uint64_std

AsyncVector_uint64: TypeAlias = PODVector_uint64_std

HostVector_uint64: TypeAlias = PODVector_uint64_std

PODVector_uint64_default: TypeAlias = PODVector_uint64_std

class TagBox:
    """
    Cell-tag storage used by ``AmrCore.error_est``.

    Use ``TagBox.SET`` to request refinement, ``TagBox.CLEAR`` to remove a tag and
    ``TagBox.BUF`` for AMReX-generated buffered tags.
    """

    class TagVal(enum.IntEnum):
        CLEAR = 0

        BUF = 1

        SET = 2

    CLEAR: TagBox.TagVal = TagVal.CLEAR

    BUF: TagBox.TagVal = TagVal.BUF

    SET: TagBox.TagVal = TagVal.SET

    def __repr__(self) -> str: ...

class TagBoxArray(FabArrayBase):
    """
    Distributed array of ``TagBox`` objects used during AMR error estimation.

    Python ``AmrCore.error_est`` overrides receive a mutable ``TagBoxArray`` and
    mark cells with ``set_val(TagBox.SET, ...)``.  Callback arguments are
    non-owning views and should not be stored after the override returns.
    """

    @overload
    def __init__(self, ba: BoxArray, dm: DistributionMapping, ngrow: int = 0) -> None:
        """Construct tag storage on ba/dm with an isotropic grow width."""

    @overload
    def __init__(self, ba: BoxArray, dm: DistributionMapping, ngrow: IntVect3D) -> None:
        """Construct tag storage on ba/dm with per-direction grow widths."""

    def __repr__(self) -> str: ...

    def clear(self) -> None:
        """Release all tag data and metadata."""

    def ok(self) -> bool:
        """Return True if the tag array is internally consistent."""

    def __len__(self) -> int: ...

    @property
    def size(self) -> int:
        """Number of boxes in the global tag layout."""

    @property
    def local_size(self) -> int:
        """Number of tag boxes owned by this MPI rank."""

    @property
    def n_grow_vect(self) -> IntVect3D:
        """Grow width of the tag storage in each coordinate direction."""

    @property
    def box_array(self) -> BoxArray:
        """BoxArray defining the valid regions for this tag array."""

    @property
    def dist_map(self) -> DistributionMapping:
        """DistributionMapping defining ownership of this tag array."""

    @overload
    def set_val(self, val: TagBox.TagVal) -> None:
        """Set all valid and grow cells in the tag array to val."""

    @overload
    def set_val(self, val: TagBox.TagVal, nghost: int) -> None:
        """Set all valid cells plus nghost grow cells to val."""

    @overload
    def set_val(self, val: TagBox.TagVal, nghost: IntVect3D) -> None:
        """Set all valid cells plus per-direction grow cells to val."""

    @overload
    def set_val(self, val: TagBox.TagVal, region: Box, nghost: int = 0) -> None:
        """Set cells intersecting region, optionally grown by nghost, to val."""

    @overload
    def set_val(self, val: TagBox.TagVal, region: Box, nghost: IntVect3D) -> None:
        """Set cells intersecting region with per-direction grow widths to val."""

    @overload
    def set_val(self, ba: BoxArray, val: TagBox.TagVal) -> None:
        """Set cells covered by ba to val."""

    def buffer(self, nbuf: IntVect3D) -> None:
        """Grow every SET tag by nbuf cells using AMReX tag-buffer rules."""

    def map_periodic_remove_duplicates(self, geom: Geometry) -> None:
        """
        Map tags through periodic boundaries described by geom and remove duplicates.
        """

    def coarsen(self, ratio: IntVect3D) -> None:
        """Coarsen tags in place by ratio."""

    def has_tags(self, box: Box) -> bool:
        """Return True if box contains any SET or BUF tags."""

class AmrInfo:
    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def verbose(self) -> int: ...

    @verbose.setter
    def verbose(self, arg: int, /) -> None: ...

    @property
    def max_level(self) -> int: ...

    @max_level.setter
    def max_level(self, arg: int, /) -> None: ...

    def ref_ratio(self, arg: int, /) -> IntVect3D: ...

    def blocking_factor(self, arg: int, /) -> IntVect3D: ...

    def max_grid_size(self, arg: int, /) -> IntVect3D: ...

    def n_error_buf(self, arg: int, /) -> IntVect3D: ...

    @property
    def grid_eff(self) -> float: ...

    @grid_eff.setter
    def grid_eff(self, arg: float, /) -> None: ...

    @property
    def n_proper(self) -> int: ...

    @n_proper.setter
    def n_proper(self, arg: int, /) -> None: ...

    @property
    def use_fixed_upto_level(self) -> int: ...

    @use_fixed_upto_level.setter
    def use_fixed_upto_level(self, arg: int, /) -> None: ...

    @property
    def use_fixed_coarse_grids(self) -> bool: ...

    @use_fixed_coarse_grids.setter
    def use_fixed_coarse_grids(self, arg: bool, /) -> None: ...

    @property
    def refine_grid_layout(self) -> bool: ...

    @refine_grid_layout.setter
    def refine_grid_layout(self, arg: bool, /) -> None: ...

    @property
    def refine_grid_layout_dims(self) -> IntVect3D: ...

    @refine_grid_layout_dims.setter
    def refine_grid_layout_dims(self, arg: IntVect3D, /) -> None: ...

    @property
    def check_input(self) -> bool: ...

    @check_input.setter
    def check_input(self, arg: bool, /) -> None: ...

    @property
    def use_new_chop(self) -> bool: ...

    @use_new_chop.setter
    def use_new_chop(self, arg: bool, /) -> None: ...

    @property
    def iterate_on_new_grids(self) -> bool: ...

    @iterate_on_new_grids.setter
    def iterate_on_new_grids(self, arg: bool, /) -> None: ...

class AmrMesh:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, rb: RealBox, max_level_in: int, n_cell_in: Vector_int, coord: int, ref_ratios: Vector_IntVect, is_per: Sequence[int]) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def verbose(self) -> int: ...

    @property
    def max_level(self) -> int: ...

    @property
    def finest_level(self) -> int: ...

    @overload
    def ref_ratio(self) -> Vector_IntVect: ...

    @overload
    def ref_ratio(self, arg: int, /) -> IntVect3D: ...

    def geom(self, lev: int) -> Geometry:
        """Return the Geometry stored for AMR level lev."""

    def set_geometry(self, lev: int, geom_in: Geometry) -> None:
        """Replace the Geometry stored for AMR level lev."""

class AmrCore(AmrMesh):
    """
    Base class for Python AMR applications that manage an AMReX mesh hierarchy.

    Subclasses must implement ``make_new_level_from_scratch``,
    ``make_new_level_from_coarse``, ``remake_level``, ``clear_level`` and
    ``error_est``.  AMReX calls these Python overrides while creating or
    regridding levels.

    ``error_est(lev, tags, time, ngrow)`` receives a mutable ``TagBoxArray``
    for the level being tagged.  Mark cells with
    ``tags.set_val(TagBox.SET, ...)`` and keep the tag array only for the
    duration of the callback.
    """

    @overload
    def __init__(self) -> None:
        """
        Construct an empty AMR core.

        The mesh metadata is read from AMReX runtime parameters when available.
        """

    @overload
    def __init__(self, rb: RealBox, max_level_in: int, n_cell_in: Vector_int, coord: int, ref_ratios: Vector_IntVect, is_per: Sequence[int]) -> None:
        """
        Construct an AMR core from an explicit level-0 problem domain.

        Parameters
        ----------
        rb : RealBox
            Physical problem domain for level 0.
        max_level_in : int
            Maximum AMR level to create.  Use 0 for a single-level hierarchy.
        n_cell_in : Vector_int
            Number of level-0 cells in each coordinate direction.
        coord : int
            AMReX coordinate-system identifier.
        ref_ratios : Vector_IntVect
            Refinement ratio for each coarse level.  Its length is normally
            ``max_level_in``.
        is_per : Sequence[int]
            Periodicity flags for each coordinate direction.
        """

    @overload
    def __init__(self, level_0_geom: Geometry, amr_info: AmrInfo) -> None:
        """
        Construct an AMR core from a level-0 geometry and an ``AmrInfo`` object.
        """

    def __repr__(self) -> str: ...

    def init_from_scratch(self, time: float) -> None:
        """
        Create the AMR hierarchy from scratch at simulation time ``time``.

        This calls the Python overrides that allocate level data and, when
        ``max_level`` is greater than 0, calls ``error_est`` to create refined grids.
        """

    def regrid(self, lbase: int, time: float, initial: bool = False) -> None:
        """
        Rebuild levels finer than ``lbase`` at simulation time ``time``.

        ``error_est`` is called to tag cells, followed by the level remake/create/clear
        callbacks as needed.
        """

    def get_par_gdb(self) -> AmrParGDB:
        """
        Return the particle geometry/database broker owned by this AMR core.

        The returned ``AmrParGDB`` can be passed to particle-container constructors or
        ``define`` methods.  It is a non-owning view; the ``AmrCore`` is kept alive by
        the binding while the broker is used from Python.
        """

class ParGDBBase:
    """
    Abstract broker for particle geometry, box arrays and distribution maps.

    Particle containers use a ``ParGDBBase`` to query mesh metadata for each AMR
    level.  Python users usually obtain a concrete ``AmrParGDB`` from
    ``AmrCore.get_par_gdb()``.
    """

    def particle_geom(self, level: int) -> Geometry:
        """Return particle Geometry for AMR level."""

    def geom(self, level: int) -> Geometry:
        """Return mesh Geometry for AMR level."""

    def particle_dist_map(self, level: int) -> DistributionMapping:
        """Return particle DistributionMapping for AMR level."""

    def dist_map(self, level: int) -> DistributionMapping:
        """Return mesh DistributionMapping for AMR level."""

    def particle_box_array(self, level: int) -> BoxArray:
        """Return particle BoxArray for AMR level."""

    def box_array(self, level: int) -> BoxArray:
        """Return mesh BoxArray for AMR level."""

    def set_particle_box_array(self, level: int, new_ba: BoxArray) -> None:
        """Replace the particle BoxArray for AMR level."""

    def set_particle_dist_map(self, level: int, new_dm: DistributionMapping) -> None:
        """Replace the particle DistributionMapping for AMR level."""

    def set_particle_geometry(self, level: int, new_geom: Geometry) -> None:
        """Replace the particle Geometry for AMR level."""

    def level_defined(self, level: int) -> bool:
        """Return True if AMR level has valid mesh metadata."""

    def finest_level(self) -> int:
        """Return the finest currently defined AMR level."""

    def max_level(self) -> int:
        """Return the maximum AMR level supported by this broker."""

    def ref_ratio(self, level: int) -> IntVect3D:
        """Return the refinement ratio from level to level + 1."""

class AmrParGDB(ParGDBBase):
    """Concrete particle metadata broker backed by an AmrCore."""

    def __init__(self, amr_core: AmrCore) -> None:
        """Construct a particle metadata broker backed by amr_core."""

class Particle_3_0:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> object: ...

    @overload
    def get_idata(self) -> object: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class StructOfArrays_3_0_idcpu_pinned:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_pinned]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_pinned]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_pinned:
        """Get access to a particle IdCPU component Array"""

class StructOfArrays_3_0_idcpu_default:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_std]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_std]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_std:
        """Get access to a particle IdCPU component Array"""

class StructOfArrays_3_0_idcpu_arena:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_arena]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_arena]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_arena:
        """Get access to a particle IdCPU component Array"""

class StructOfArrays_3_0_idcpu_polymorphic:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_polymorphic]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_polymorphic]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_polymorphic:
        """Get access to a particle IdCPU component Array"""

class ParticleTileData_pureSoA_3_0:
    def __init__(self) -> None: ...

    @property
    def m_size(self) -> int: ...

    @property
    def m_num_runtime_real(self) -> int: ...

    @property
    def m_num_runtime_int(self) -> int: ...

    def get_super_particle(self, arg: int, /) -> Particle_3_0: ...

    def set_super_particle(self, arg0: Particle_3_0, arg1: int, /) -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_3_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_3_0: ...

class ParticleTile_pureSoA_3_0_pinned:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_0_idcpu_pinned: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_3_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_3_0_pinned, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_3_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_3_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_3_0: ...

class ParticleTile_pureSoA_3_0_default:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_0_idcpu_default: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_3_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_3_0_default, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_3_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_3_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_3_0: ...

class ParticleTile_pureSoA_3_0_arena:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_0_idcpu_arena: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_3_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_3_0_arena, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_3_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_3_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_3_0: ...

class ParticleTile_pureSoA_3_0_polymorphic:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_0_idcpu_polymorphic: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_3_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_3_0_polymorphic, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_3_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_3_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_3_0: ...

class ParticleInitType_pureSoA_3_0:
    def __init__(self) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def real_array_data(self) -> list[float]: ...

    @real_array_data.setter
    def real_array_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_array_data(self) -> list[int]: ...

    @int_array_data.setter
    def int_array_data(self, arg: Sequence[int], /) -> None: ...

class ParIterBase_pureSoA_3_0_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_pinned: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_3_0_pinned(ParIterBase_pureSoA_3_0_pinned):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_3_0_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_pinned: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_3_0_pinned(ParConstIterBase_pureSoA_3_0_pinned):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_3_0_pinned:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_3_0_pinned: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_3_0_pinned, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_3_0_pinned, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_3_0_pinned]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_3_0_pinned:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_3_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_pureSoA_3_0_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_default: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_3_0_default(ParIterBase_pureSoA_3_0_default):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_3_0_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_default: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_3_0_default(ParConstIterBase_pureSoA_3_0_default):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_3_0_default:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_3_0_default: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_3_0_default, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_3_0_default, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_3_0_default]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_3_0_default:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_3_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_pureSoA_3_0_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_arena: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_3_0_arena(ParIterBase_pureSoA_3_0_arena):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_3_0_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_arena: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_3_0_arena(ParConstIterBase_pureSoA_3_0_arena):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_3_0_arena:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_3_0_arena: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_3_0_arena, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_3_0_arena, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_3_0_arena]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_3_0_arena:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_3_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_pureSoA_3_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_polymorphic: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_3_0_polymorphic(ParIterBase_pureSoA_3_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_3_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_3_0_polymorphic: ...

    def soa(self) -> StructOfArrays_3_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_3_0_polymorphic(ParConstIterBase_pureSoA_3_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_3_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_3_0_polymorphic:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_3_0_polymorphic: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_3_0_polymorphic, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_3_0_polymorphic, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_3_0_polymorphic]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_3_0_polymorphic:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_3_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class Particle_2_1:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> int: ...

    @overload
    def get_idata(self) -> list[int]: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class Particle_5_2:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> int: ...

    @overload
    def get_idata(self) -> list[int]: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class ArrayOfStructs_2_1_pinned:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_2_1, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_2_1:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_2_1, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_2_1: ...

    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...

class ArrayOfStructs_2_1_default:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_2_1, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_2_1:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_2_1, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_2_1: ...

    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...

class ArrayOfStructs_2_1_arena:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_2_1, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_2_1:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_2_1, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_2_1: ...

    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...

class ArrayOfStructs_2_1_polymorphic:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_2_1, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_2_1:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_2_1, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_2_1: ...

    def to_host(self) -> ArrayOfStructs_2_1_pinned: ...

class StructOfArrays_3_1_pinned:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_pinned]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_pinned]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_3_1_default:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_std]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_std]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_3_1_arena:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_arena]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_arena]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_3_1_polymorphic:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_polymorphic]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_polymorphic]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class ParticleTileData_2_1_3_1:
    def __init__(self) -> None: ...

    @property
    def m_size(self) -> int: ...

    @property
    def m_num_runtime_real(self) -> int: ...

    @property
    def m_num_runtime_int(self) -> int: ...

    def get_super_particle(self, arg: int, /) -> Particle_5_2: ...

    def set_super_particle(self, arg0: Particle_5_2, arg1: int, /) -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_5_2, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_5_2: ...

class ParticleTile_2_1_3_1_pinned:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_1_pinned: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_2_1, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_5_2, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_2_1_3_1_pinned, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...

    def __setitem__(self, arg0: int, arg1: Particle_5_2, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_5_2: ...

    def get_array_of_structs(self) -> ArrayOfStructs_2_1_pinned: ...

class ParticleTile_2_1_3_1_default:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_1_default: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_2_1, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_5_2, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_2_1_3_1_default, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...

    def __setitem__(self, arg0: int, arg1: Particle_5_2, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_5_2: ...

    def get_array_of_structs(self) -> ArrayOfStructs_2_1_default: ...

class ParticleTile_2_1_3_1_arena:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_1_arena: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_2_1, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_5_2, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_2_1_3_1_arena, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...

    def __setitem__(self, arg0: int, arg1: Particle_5_2, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_5_2: ...

    def get_array_of_structs(self) -> ArrayOfStructs_2_1_arena: ...

class ParticleTile_2_1_3_1_polymorphic:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_3_1_polymorphic: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_2_1, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_5_2, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_2_1_3_1_polymorphic, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_2_1_3_1: ...

    def __setitem__(self, arg0: int, arg1: Particle_5_2, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_5_2: ...

    def get_array_of_structs(self) -> ArrayOfStructs_2_1_polymorphic: ...

class ParticleInitType_2_1_3_1:
    def __init__(self) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def real_array_data(self) -> list[float]: ...

    @real_array_data.setter
    def real_array_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_array_data(self) -> list[int]: ...

    @int_array_data.setter
    def int_array_data(self, arg: Sequence[int], /) -> None: ...

    @property
    def real_struct_data(self) -> list[float]: ...

    @real_struct_data.setter
    def real_struct_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_struct_data(self) -> list[int]: ...

    @int_struct_data.setter
    def int_struct_data(self, arg: Sequence[int], /) -> None: ...

class ParIterBase_2_1_3_1_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_pinned: ...

    def soa(self) -> StructOfArrays_3_1_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_pinned: ...

class ParIter_2_1_3_1_pinned(ParIterBase_2_1_3_1_pinned):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_2_1_3_1_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_pinned: ...

    def soa(self) -> StructOfArrays_3_1_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_pinned: ...

class ParConstIter_2_1_3_1_pinned(ParConstIterBase_2_1_3_1_pinned):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_2_1_3_1_pinned:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_2_1_3_1_pinned: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_2_1_3_1_pinned, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_2_1_3_1_pinned, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_2_1_3_1_pinned]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_2_1_3_1_pinned:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_2_1_3_1, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_2_1_3_1_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_default: ...

    def soa(self) -> StructOfArrays_3_1_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_default: ...

class ParIter_2_1_3_1_default(ParIterBase_2_1_3_1_default):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_2_1_3_1_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_default: ...

    def soa(self) -> StructOfArrays_3_1_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_default: ...

class ParConstIter_2_1_3_1_default(ParConstIterBase_2_1_3_1_default):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_2_1_3_1_default:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_2_1_3_1_default: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_2_1_3_1_default, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_2_1_3_1_default, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_2_1_3_1_default]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_2_1_3_1_default:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_2_1_3_1, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_2_1_3_1_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_arena: ...

    def soa(self) -> StructOfArrays_3_1_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_arena: ...

class ParIter_2_1_3_1_arena(ParIterBase_2_1_3_1_arena):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_2_1_3_1_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_arena: ...

    def soa(self) -> StructOfArrays_3_1_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_arena: ...

class ParConstIter_2_1_3_1_arena(ParConstIterBase_2_1_3_1_arena):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_2_1_3_1_arena:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_2_1_3_1_arena: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_2_1_3_1_arena, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_2_1_3_1_arena, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_2_1_3_1_arena]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_2_1_3_1_arena:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_2_1_3_1, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_2_1_3_1_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_polymorphic: ...

    def soa(self) -> StructOfArrays_3_1_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_polymorphic: ...

class ParIter_2_1_3_1_polymorphic(ParIterBase_2_1_3_1_polymorphic):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_2_1_3_1_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_2_1_3_1_polymorphic: ...

    def soa(self) -> StructOfArrays_3_1_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_2_1_polymorphic: ...

class ParConstIter_2_1_3_1_polymorphic(ParConstIterBase_2_1_3_1_polymorphic):
    def __init__(self, particle_container: ParticleContainer_2_1_3_1_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_2_1_3_1_polymorphic:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_2_1_3_1_polymorphic: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_2_1_3_1_polymorphic, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_2_1_3_1_polymorphic, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_2_1_3_1_polymorphic]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_2_1_3_1_polymorphic:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_2_1_3_1, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_2_1_3_1, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class Particle_16_4:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> int: ...

    @overload
    def get_idata(self) -> list[int]: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class ArrayOfStructs_16_4_pinned:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_16_4, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_16_4:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...

class ArrayOfStructs_16_4_default:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_16_4, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_16_4:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...

class ArrayOfStructs_16_4_arena:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_16_4, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_16_4:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...

class ArrayOfStructs_16_4_polymorphic:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def numParticles(self) -> int: ...

    def numRealParticles(self) -> int: ...

    def numNeighborParticles(self) -> int: ...

    def numTotalParticles(self) -> int: ...

    def setNumNeighbors(self, arg: int, /) -> None: ...

    def getNumNeighbors(self) -> int: ...

    @overload
    def empty(self) -> bool: ...

    @overload
    def empty(self) -> bool: ...

    def push_back(self, arg: Particle_16_4, /) -> None: ...

    def pop_back(self) -> None: ...

    def back(self) -> Particle_16_4:
        """get back member.  Problem!!!!! this is perfo"""

    @property
    def __array_interface__(self) -> dict: ...

    @property
    def __cuda_array_interface__(self) -> dict: ...

    def test_sizes() -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def to_host(self) -> ArrayOfStructs_16_4_pinned: ...

class StructOfArrays_0_0_pinned:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_pinned]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_pinned]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_pinned:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_0_0_default:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_std]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_std]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_std:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_0_0_arena:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_arena]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_arena]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_arena:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class StructOfArrays_0_0_polymorphic:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_polymorphic]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_polymorphic]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

class ParticleTileData_16_4_0_0:
    def __init__(self) -> None: ...

    @property
    def m_size(self) -> int: ...

    @property
    def m_num_runtime_real(self) -> int: ...

    @property
    def m_num_runtime_int(self) -> int: ...

    def get_super_particle(self, arg: int, /) -> Particle_16_4: ...

    def set_super_particle(self, arg0: Particle_16_4, arg1: int, /) -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

class ParticleTile_16_4_0_0_pinned:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_0_0_pinned: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_16_4_0_0_pinned, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def get_array_of_structs(self) -> ArrayOfStructs_16_4_pinned: ...

class ParticleTile_16_4_0_0_default:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_0_0_default: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_16_4_0_0_default, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def get_array_of_structs(self) -> ArrayOfStructs_16_4_default: ...

class ParticleTile_16_4_0_0_arena:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_0_0_arena: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_16_4_0_0_arena, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def get_array_of_structs(self) -> ArrayOfStructs_16_4_arena: ...

class ParticleTile_16_4_0_0_polymorphic:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_0_0_polymorphic: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back(self, arg: Particle_16_4, /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_16_4_0_0_polymorphic, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_16_4_0_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_16_4, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_16_4: ...

    def get_array_of_structs(self) -> ArrayOfStructs_16_4_polymorphic: ...

class ParticleInitType_16_4_0_0:
    def __init__(self) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def real_array_data(self) -> list[float]: ...

    @real_array_data.setter
    def real_array_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_array_data(self) -> list[int]: ...

    @int_array_data.setter
    def int_array_data(self, arg: Sequence[int], /) -> None: ...

    @property
    def real_struct_data(self) -> list[float]: ...

    @real_struct_data.setter
    def real_struct_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_struct_data(self) -> list[int]: ...

    @int_struct_data.setter
    def int_struct_data(self, arg: Sequence[int], /) -> None: ...

class ParIterBase_16_4_0_0_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_pinned: ...

    def soa(self) -> StructOfArrays_0_0_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_pinned: ...

class ParIter_16_4_0_0_pinned(ParIterBase_16_4_0_0_pinned):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_16_4_0_0_pinned(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_pinned, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_pinned: ...

    def soa(self) -> StructOfArrays_0_0_pinned: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_pinned: ...

class ParConstIter_16_4_0_0_pinned(ParConstIterBase_16_4_0_0_pinned):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_pinned, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_16_4_0_0_pinned:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_16_4_0_0_pinned: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_16_4_0_0_pinned, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_16_4_0_0_pinned, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_16_4_0_0_pinned]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_16_4_0_0_pinned:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_16_4_0_0, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_16_4_0_0_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_default: ...

    def soa(self) -> StructOfArrays_0_0_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_default: ...

class ParIter_16_4_0_0_default(ParIterBase_16_4_0_0_default):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_16_4_0_0_default(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_default, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_default: ...

    def soa(self) -> StructOfArrays_0_0_default: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_default: ...

class ParConstIter_16_4_0_0_default(ParConstIterBase_16_4_0_0_default):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_default, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_16_4_0_0_default:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_16_4_0_0_default: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_16_4_0_0_default, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_16_4_0_0_default, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_16_4_0_0_default]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_16_4_0_0_default:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_16_4_0_0, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_16_4_0_0_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_arena: ...

    def soa(self) -> StructOfArrays_0_0_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_arena: ...

class ParIter_16_4_0_0_arena(ParIterBase_16_4_0_0_arena):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_16_4_0_0_arena(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_arena, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_arena: ...

    def soa(self) -> StructOfArrays_0_0_arena: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_arena: ...

class ParConstIter_16_4_0_0_arena(ParConstIterBase_16_4_0_0_arena):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_arena, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_16_4_0_0_arena:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_16_4_0_0_arena: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_16_4_0_0_arena, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_16_4_0_0_arena, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_16_4_0_0_arena]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_16_4_0_0_arena:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_16_4_0_0, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class ParIterBase_16_4_0_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_polymorphic: ...

    def soa(self) -> StructOfArrays_0_0_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_polymorphic: ...

class ParIter_16_4_0_0_polymorphic(ParIterBase_16_4_0_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_16_4_0_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_16_4_0_0_polymorphic: ...

    def soa(self) -> StructOfArrays_0_0_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

    def aos(self) -> ArrayOfStructs_16_4_polymorphic: ...

class ParConstIter_16_4_0_0_polymorphic(ParConstIterBase_16_4_0_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_16_4_0_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_16_4_0_0_polymorphic:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_16_4_0_0_polymorphic: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_16_4_0_0_polymorphic, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_16_4_0_0_polymorphic, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_16_4_0_0_polymorphic]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_16_4_0_0_polymorphic:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, arg3: bool, arg4: RealBox, /) -> None: ...

    def init_random_per_box(self, arg0: int, arg1: int, arg2: ParticleInitType_16_4_0_0, /) -> None: ...

    def init_one_per_cell(self, arg0: float, arg1: float, arg2: float, arg3: ParticleInitType_16_4_0_0, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class Particle_11_0:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> object: ...

    @overload
    def get_idata(self) -> object: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class StructOfArrays_11_0_idcpu_polymorphic:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_polymorphic]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_polymorphic]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_polymorphic:
        """Get access to a particle IdCPU component Array"""

class ParticleTileData_pureSoA_11_0:
    def __init__(self) -> None: ...

    @property
    def m_size(self) -> int: ...

    @property
    def m_num_runtime_real(self) -> int: ...

    @property
    def m_num_runtime_int(self) -> int: ...

    def get_super_particle(self, arg: int, /) -> Particle_11_0: ...

    def set_super_particle(self, arg0: Particle_11_0, arg1: int, /) -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_11_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_11_0: ...

class ParticleTile_pureSoA_11_0_polymorphic:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_11_0_idcpu_polymorphic: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_11_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_11_0_polymorphic, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_11_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_11_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_11_0: ...

class ParticleInitType_pureSoA_11_0:
    def __init__(self) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def real_array_data(self) -> list[float]: ...

    @real_array_data.setter
    def real_array_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_array_data(self) -> list[int]: ...

    @int_array_data.setter
    def int_array_data(self, arg: Sequence[int], /) -> None: ...

class ParIterBase_pureSoA_11_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_11_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_11_0_polymorphic: ...

    def soa(self) -> StructOfArrays_11_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_11_0_polymorphic(ParIterBase_pureSoA_11_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_11_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_11_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_11_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_11_0_polymorphic: ...

    def soa(self) -> StructOfArrays_11_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_11_0_polymorphic(ParConstIterBase_pureSoA_11_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_11_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_11_0_polymorphic:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_11_0_polymorphic: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_11_0_polymorphic, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_11_0_polymorphic, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_11_0_polymorphic]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_11_0_polymorphic:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_11_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

class Particle_7_0:
    @overload
    def __init__(self, **kwargs) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, *args) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, /, **kwargs) -> None: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    NReal: int = ...
    """(arg: object, /) -> int"""

    NInt: int = ...
    """(arg: object, /) -> int"""

    @overload
    def pos(self, arg: int, /) -> float: ...

    @overload
    def pos(self) -> RealVect: ...

    @overload
    def setPos(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def setPos(self, arg: RealVect, /) -> None: ...

    @overload
    def setPos(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_rdata(self, arg: int, /) -> float: ...

    @overload
    def get_rdata(self) -> list[float]: ...

    @overload
    def set_rdata(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def set_rdata(self, arg: Sequence[float], /) -> None: ...

    @overload
    def get_idata(self, arg: int, /) -> object: ...

    @overload
    def get_idata(self) -> object: ...

    @overload
    def set_idata(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def set_idata(self, arg: Sequence[int], /) -> None: ...

    def cpu(self) -> int: ...

    def id(self) -> int: ...

    @overload
    def NextID(self) -> int: ...

    @overload
    def NextID(self, arg: int, /) -> None: ...

    @property
    def x(self) -> float: ...

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float: ...

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float: ...

    @z.setter
    def z(self, arg: float, /) -> None: ...

class StructOfArrays_7_0_idcpu_polymorphic:
    def __init__(self) -> None: ...

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], /) -> None: ...

    @property
    def num_real_comps(self) -> int:
        """Get the number of compile-time + runtime Real components"""

    @property
    def num_int_comps(self) -> int:
        """Get the number of compile-time + runtime Int components"""

    @property
    def has_idcpu(self) -> bool:
        """In pure SoA particle layout, idcpu is an array in the SoA"""

    @overload
    def get_real_data(self) -> list[PODVector_real_polymorphic]:
        """Get access to the particle Real Arrays (only compile-time components)"""

    @overload
    def get_real_data(self, index: int) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_real_data(self, name: str) -> PODVector_real_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self) -> list[PODVector_int_polymorphic]:
        """Get access to the particle Int Arrays (only compile-time components)"""

    @overload
    def get_int_data(self, index: int) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @overload
    def get_int_data(self, name: str) -> PODVector_int_polymorphic:
        """
        Get access to a particle Real component Array (compile-time and runtime component)
        """

    @property
    def real_names(self) -> list[str]:
        """Names for the Real SoA components"""

    @property
    def int_names(self) -> list[str]:
        """Names for the int SoA components"""

    def __len__(self) -> int:
        """Get the number of particles"""

    @property
    def size(self) -> int:
        """Get the number of particles"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, new_size: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def get_idcpu_data(self) -> PODVector_uint64_polymorphic:
        """Get access to a particle IdCPU component Array"""

class ParticleTileData_pureSoA_7_0:
    def __init__(self) -> None: ...

    @property
    def m_size(self) -> int: ...

    @property
    def m_num_runtime_real(self) -> int: ...

    @property
    def m_num_runtime_int(self) -> int: ...

    def get_super_particle(self, arg: int, /) -> Particle_7_0: ...

    def set_super_particle(self, arg0: Particle_7_0, arg1: int, /) -> None: ...

    def __setitem__(self, arg0: int, arg1: Particle_7_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_7_0: ...

class ParticleTile_pureSoA_7_0_polymorphic:
    def __init__(self) -> None: ...

    NAR: int = ...
    """(arg: object, /) -> int"""

    NAI: int = ...
    """(arg: object, /) -> int"""

    def define(self, arg0: int, arg1: int, arg2: Sequence[str], arg3: Sequence[str], arg4: Arena, /) -> None: ...

    def get_struct_of_arrays(self) -> StructOfArrays_7_0_idcpu_polymorphic: ...

    @property
    def empty(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def num_total_particles(self) -> int: ...

    def set_num_neighbors(self, arg: int, /) -> None: ...

    def get_num_neighbors(self) -> int: ...

    def resize(self, count: int, strategy: GrowthStrategy = GrowthStrategy.Poisson) -> None: ...

    def push_back(self, arg: Particle_7_0, /) -> None:
        """Add one particle to this tile."""

    @overload
    def push_back_real(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def push_back_real(self, arg: Sequence[float], /) -> None: ...

    @overload
    def push_back_real(self, arg0: int, arg1: int, arg2: float, /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def push_back_int(self, arg: Sequence[int], /) -> None: ...

    @overload
    def push_back_int(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def num_real_comps(self) -> int: ...

    @property
    def num_int_comps(self) -> int: ...

    @property
    def num_runtime_real_comps(self) -> int: ...

    @property
    def num_runtime_int_comps(self) -> int: ...

    def shrink_to_fit(self) -> None: ...

    def capacity(self) -> int: ...

    def swap(self, arg: ParticleTile_pureSoA_7_0_polymorphic, /) -> None: ...

    def get_particle_tile_data(self) -> ParticleTileData_pureSoA_7_0: ...

    def __setitem__(self, arg0: int, arg1: Particle_7_0, /) -> None: ...

    def __getitem__(self, arg: int, /) -> Particle_7_0: ...

class ParticleInitType_pureSoA_7_0:
    def __init__(self) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def real_array_data(self) -> list[float]: ...

    @real_array_data.setter
    def real_array_data(self, arg: Sequence[float], /) -> None: ...

    @property
    def int_array_data(self) -> list[int]: ...

    @int_array_data.setter
    def int_array_data(self, arg: Sequence[int], /) -> None: ...

class ParIterBase_pureSoA_7_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_7_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_7_0_polymorphic: ...

    def soa(self) -> StructOfArrays_7_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParIter_pureSoA_7_0_polymorphic(ParIterBase_pureSoA_7_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_7_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParConstIterBase_pureSoA_7_0_polymorphic(MFIter):
    def __init__(self, particle_container: ParticleContainer_pureSoA_7_0_polymorphic, level: int) -> None: ...

    def particle_tile(self) -> ParticleTile_pureSoA_7_0_polymorphic: ...

    def soa(self) -> StructOfArrays_7_0_idcpu_polymorphic: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    @property
    def size(self) -> int:
        """the number of particles on this tile"""

    @property
    def num_particles(self) -> int: ...

    @property
    def num_real_particles(self) -> int: ...

    @property
    def num_neighbor_particles(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def pair_index(self) -> tuple[int, int]: ...

    @property
    def is_valid(self) -> bool: ...

    def geom(self, level: int) -> Geometry: ...

    def finalize(self) -> None: ...

class ParConstIter_pureSoA_7_0_polymorphic(ParConstIterBase_pureSoA_7_0_polymorphic):
    def __init__(self, particle_container: ParticleContainer_pureSoA_7_0_polymorphic, level: int) -> None: ...

    def __repr__(self) -> str: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

class ParticleContainer_pureSoA_7_0_polymorphic:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def __init__(self, gdb: ParGDBBase) -> None:
        """
        Construct from a particle metadata broker such as AmrCore.get_par_gdb().
        """

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def __init__(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def define(self, gdb: ParGDBBase) -> None:
        """Define this container from a particle metadata broker."""

    def make_alike(self) -> ParticleContainer_pureSoA_7_0_polymorphic: ...

    @property
    def arena(self) -> Arena: ...

    @arena.setter
    def arena(self, arg: Arena, /) -> None: ...

    is_soa_particle: bool = ...
    """(arg: object, /) -> bool"""

    num_struct_real: int = ...
    """(arg: object, /) -> int"""

    num_struct_int: int = ...
    """(arg: object, /) -> int"""

    num_array_real: int = ...
    """(arg: object, /) -> int"""

    num_array_int: int = ...
    """(arg: object, /) -> int"""

    @property
    def num_real_comps(self) -> int:
        """The number of compile-time and runtime Real components in SoA"""

    @property
    def num_int_comps(self) -> int:
        """The number of compile-time and runtime int components in SoA"""

    @property
    def num_runtime_real_comps(self) -> int:
        """The number of runtime Real components in SoA"""

    @property
    def num_runtime_int_comps(self) -> int:
        """The number of runtime Int components in SoA"""

    @property
    def num_position_components(self) -> int: ...

    @property
    def byte_spread(self) -> list[int]: ...

    def set_soa_compile_time_names(self, arg0: Sequence[str], arg1: Sequence[str], /) -> None: ...

    @overload
    def add_real_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Real"""

    @overload
    def add_real_comp(self, name: str, communicate: int = 1) -> None: ...

    @overload
    def add_int_comp(self, communicate: int = 1) -> None:
        """add a new runtime component with type Int"""

    @overload
    def add_int_comp(self, name: str, communicate: int = 1) -> None: ...

    @property
    def real_soa_names(self) -> list[str]:
        """Get the names for the Real SoA components"""

    @property
    def int_soa_names(self) -> list[str]:
        """Get the names for the int SoA components"""

    def has_real_comp(self, arg: str, /) -> bool:
        """Check if a container has an ParticleReal component"""

    def has_int_comp(self, arg: str, /) -> bool:
        """Check if a container has an Integer component"""

    def get_real_comp_index(self, arg: str, /) -> int:
        """Get the ParticleReal SoA index of a component"""

    def get_int_comp_index(self, arg: str, /) -> int:
        """Get the Integer SoA index of a component"""

    @property
    def finest_level(self) -> int: ...

    @overload
    def Define(self, arg0: Geometry, arg1: DistributionMapping, arg2: BoxArray, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_int, /) -> None: ...

    @overload
    def Define(self, arg0: Vector_Geometry, arg1: Vector_DistributionMapping, arg2: Vector_BoxArray, arg3: Vector_IntVect, /) -> None: ...

    def num_local_tiles_at_level(self, level: int) -> int: ...

    def reserve_data(self) -> None: ...

    def resize_data(self) -> None: ...

    def increment(self, arg0: MultiFab, arg1: int, /) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0, local: int = 0, remove_negative: bool = True) -> None: ...

    @overload
    def redistribute(self, lev_min: int = 0, lev_max: int = -1, nGrow: IntVect3D = ..., local: bool = False, max_cells_moved: IntVect3D = ..., remove_negative: bool = True) -> None: ...

    def sort_particles_by_cell(self) -> None: ...

    def sort_particles_by_bin(self, arg: IntVect3D, /) -> None: ...

    def OK(self, lev_min: int = 0, lev_max: int = -1, nGrow: int = 0) -> bool: ...

    def print_capacity(self) -> list[int]: ...

    def shrink_t_fit(self) -> None: ...

    def number_of_particles_at_level(self, level: int, only_valid: bool = True, only_local: bool = False) -> int: ...

    def number_of_particles_in_grid(self, level: int, only_valid: bool = True, only_local: bool = False) -> Vector_Long: ...

    def number_of_particles(self, only_local: bool = False) -> int:
        """
        Return the number of valid particles on all MPI ranks, unless only_local is specified.
        """

    def total_number_of_particles(self, only_valid: bool = True, only_local: bool = False) -> int:
        """
        Return the number of particles (only valid or including invalid) on all MPI ranks, unless only_local is specified.
        """

    @property
    def size(self) -> int:
        """Return the number of valid particles on all MPI ranks"""

    def remove_particles_at_level(self, arg: int, /) -> None: ...

    def remove_particles_not_at_finestLevel(self) -> None: ...

    def add_particles_at_level(self, particles: ParticleTile_pureSoA_7_0_polymorphic, level: int, ngrow: int = 0) -> None: ...

    def clear_particles(self) -> None: ...

    def add_particles(self, other: ParticleContainer_pureSoA_7_0_polymorphic, local: bool = False) -> None: ...

    def restart(self, dir: str, file: str) -> None: ...

    def restart_checkpoint(self, dir: str, file: str, is_checkpoint: bool) -> None: ...

    def write_plotfile(self, dir: str, name: str) -> None: ...

    def get_particles(self, level: int) -> dict[tuple[int, int], ParticleTile_pureSoA_7_0_polymorphic]: ...

    def define_and_return_particle_tile(self, lev: int, grid: int, tile: int) -> ParticleTile_pureSoA_7_0_polymorphic:
        """
        Define, if necessary, and return the particle tile at ``(lev, grid, tile)``.

        This is useful when particles are inserted in place into a known AMR tile.
        The returned tile is owned by the particle container.
        """

    def init_random(self, arg0: int, arg1: int, arg2: ParticleInitType_pureSoA_7_0, arg3: bool, arg4: RealBox, /) -> None: ...

    Iterator: object = ...
    """amrex iterator for particle boxes"""

    ConstIterator: object = ...
    """amrex constant iterator for particle boxes (read-only)"""

def pack_ids(arg0: Annotated[NDArray[numpy.uint64], dict(shape=(None,))], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,))], /) -> object: ...

def pack_cpus(arg0: Annotated[NDArray[numpy.uint64], dict(shape=(None,))], arg1: Annotated[NDArray[numpy.int32], dict(shape=(None,))], /) -> object: ...

@overload
def unpack_ids(arg: int, /) -> int: ...

@overload
def unpack_ids(arg: Annotated[NDArray[numpy.uint64], dict(order='C')], /) -> object: ...

@overload
def unpack_cpus(arg: int, /) -> int: ...

@overload
def unpack_cpus(arg: Annotated[NDArray[numpy.uint64], dict(order='C')], /) -> object: ...

def make_invalid(arg: int, /) -> int: ...

def make_valid(arg: int, /) -> int: ...

def is_valid(arg: int, /) -> bool: ...

def write_single_level_plotfile(plotfilename: str, mf: MultiFab, varnames: Vector_string, geom: Geometry, time: float, level_step: int, versionName: str = 'HyperCLaw-V1.1', levelPrefix: str = 'Level_', mfPrefix: str = 'Cell', extra_dirs: Vector_string = ...) -> None:
    """Writes single level plotfile"""

class PlotFileData:
    def __init__(self, arg: str, /) -> None: ...

    def spaceDim(self) -> int: ...

    def time(self) -> float: ...

    def finestLevel(self) -> int: ...

    def refRatio(self, arg: int, /) -> int: ...

    def levelStep(self, arg: int, /) -> int: ...

    def boxArray(self, arg: int, /) -> BoxArray: ...

    def DistributionMap(self, arg: int, /) -> DistributionMapping: ...

    @overload
    def syncDistributionMap(self, arg: PlotFileData, /) -> None: ...

    @overload
    def syncDistributionMap(self, arg0: int, arg1: PlotFileData, /) -> None: ...

    def coordSys(self) -> int: ...

    def probDomain(self, arg: int, /) -> Box: ...

    def probSize(self) -> list[float]: ...

    def probLo(self) -> list[float]: ...

    def probHi(self) -> list[float]: ...

    def cellSize(self, arg: int, /) -> list[float]: ...

    def varNames(self) -> Vector_string: ...

    def nComp(self) -> int: ...

    def nGrowVect(self, arg: int, /) -> IntVect3D: ...

    @overload
    def get(self, arg: int, /) -> MultiFab: ...

    @overload
    def get(self, arg0: int, arg1: str, /) -> MultiFab: ...

def concatenate(root: str, num: int, mindigits: int = 5) -> str:
    """Builds plotfile name"""

class VisMF:
    @staticmethod
    def Write(mf: FabArray_FArrayBox, name: str) -> int:
        """Writes a Multifab to the specified file"""

    @overload
    @staticmethod
    def Read(name: str) -> MultiFab:
        """Reads a MultiFab from the specified file"""

    @overload
    @staticmethod
    def Read(name: str, mf: MultiFab) -> None:
        """
        Reads a MultiFab from the specified file into the given MultiFab. The BoxArray on the disk must match the BoxArray * in mf
        """

def EB2_Build(geom: Geometry, required_coarsening_level: int, max_coarsening_level: int, ngrow: int = 4, build_coarse_level_by_coarsening: bool = True, extend_domain_face: bool = True, num_coarsen_opt: int = 0) -> None:
    """EB generation"""

class EBFArrayBoxFactory(FabFactory_FArrayBox):
    def getVolFrac(self) -> MultiFab:
        """Return volume faction MultiFab"""

class EBSupport(enum.Enum):
    basic = 1

    volume = 2

    full = 3

basic: EBSupport = EBSupport.basic

volume: EBSupport = EBSupport.volume

full: EBSupport = EBSupport.full

def makeEBFabFactory(geom: Geometry, ba: BoxArray, dm: DistributionMapping, ngrow: Vector_int, support: EBSupport) -> EBFArrayBoxFactory:
    """
    Make EBFArrayBoxFactory for given Geometry, BoxArray and DistributionMapping
    """

__author__: str = ...

__license__: str = 'BSD-3-Clause-LBNL'
