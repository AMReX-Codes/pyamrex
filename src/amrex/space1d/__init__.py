# -*- coding: utf-8 -*-

from .._dll import add_windows_dll_directories

# Register dependent DLL locations for the C++ AMReX library and potential
# shared library dependencies before importing pybind.
add_windows_dll_directories(__file__)

# import core bindings to C++
from . import amrex_1d_pybind
from .amrex_1d_pybind import *  # noqa

__version__ = amrex_1d_pybind.__version__
__doc__ = amrex_1d_pybind.__doc__
__license__ = amrex_1d_pybind.__license__
__author__ = amrex_1d_pybind.__author__


# at this place we can enhance Python classes with additional methods written
# in pure Python or add some other Python logic
#
def d_decl(x, y, z):
    """Return a tuple of the first passed element"""
    return (x,)


def Print(*args, **kwargs):
    """Wrap amrex::Print() - only the IO processor writes"""
    if not initialized():  # noqa
        print("warning: Print all - AMReX not initialized")
        print(*args, **kwargs)
    elif ParallelDescriptor.IOProcessor():  # noqa
        print(*args, **kwargs)


from ..extensions.Array4 import register_Array4_extension
from ..extensions.ArrayOfStructs import register_AoS_extension
from ..extensions.MultiFab import register_MultiFab_extension
from ..extensions.ParticleContainer import register_ParticleContainer_extension
from ..extensions.PODVector import register_PODVector_extension
from ..extensions.SmallMatrix import register_SmallMatrix_extension
from ..extensions.StructOfArrays import register_SoA_extension

register_Array4_extension(amrex_1d_pybind)
register_MultiFab_extension(amrex_1d_pybind)
register_PODVector_extension(amrex_1d_pybind)
register_SmallMatrix_extension(amrex_1d_pybind)
register_SoA_extension(amrex_1d_pybind)
register_AoS_extension(amrex_1d_pybind)
register_ParticleContainer_extension(amrex_1d_pybind)


from ..extensions.ParticleContainer import list_particle_species  # noqa
from ..extensions.ParticleContainer import read_particles as _read_particles


def read_particles(
    plotfile, particle_dir="particles", communicate=True, container=None
):
    """Read AMReX particle data from a plotfile/checkpoint into a container.

    See :py:func:`amrex.extensions.ParticleContainer.read_particles` for details.
    """
    return _read_particles(
        amrex_1d_pybind, plotfile, particle_dir, communicate, container
    )


def __getattr__(name):
    """Resolve ``xp`` lazily (PEP 562).

    ``amr.xp`` is the array namespace matching this build: NumPy on CPU, CuPy
    for CUDA/HIP, dpnp for SYCL. It is the module counterpart of the ``to_xp``
    methods, for code that needs to call into the array library itself, e.g.
    ``amr.xp.sin(...)``.

    Like every other CuPy/dpnp use in pyAMReX, those are optional dependencies:
    they are imported here on first access, never at import time, so ``import
    amrex`` works on a GPU build without them. Only touching ``amr.xp`` (or a
    ``to_cupy``/``to_dpnp``/``to_xp`` call) requires one to be installed.

    Raises
    ------
    ImportError
        On a GPU build whose array library (CuPy or dpnp) is not installed.
    """
    if name == "xp":
        import importlib

        from ..extensions.dlpack_helpers import xp_module_name

        module_name = xp_module_name(amrex_1d_pybind)
        try:
            xp = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"amrex.xp needs {module_name!r}, which is an optional "
                f"dependency of pyAMReX and is not installed. Install it, or "
                f"use the to_numpy()/to_cupy()/to_dpnp() methods directly."
            ) from e
        globals()["xp"] = xp  # subsequent lookups skip __getattr__
        return xp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
