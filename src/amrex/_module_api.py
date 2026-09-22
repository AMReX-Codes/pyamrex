"""
This file is part of pyAMReX

Copyright 2026 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

import functools
import sys
import types

from .extensions.Array4 import register_Array4_extension
from .extensions.ArrayOfStructs import register_AoS_extension
from .extensions.MultiFab import register_MultiFab_extension
from .extensions.ParticleContainer import (
    list_particle_species,
    read_particles,
    register_ParticleContainer_extension,
)
from .extensions.PODVector import register_PODVector_extension
from .extensions.SmallMatrix import register_SmallMatrix_extension
from .extensions.StructOfArrays import register_SoA_extension


def setup_module(ns, amr):
    """Populate an ``amrex.space{1,2,3}d`` namespace.

    Those three packages are identical apart from the compiled pybind module
    they wrap and their ``d_decl()``, so everything else is defined once here
    and installed into their namespace.

    Class-level additions could equally be done from the ``register_*``
    functions, because a class object is shared. Module-level names cannot:
    ``from .amrex_?d_pybind import *`` has already run by then, so a name added
    to the pybind module afterwards would not appear in the package namespace.
    They have to be written into ``ns`` instead, which is what this does.

    Injected callables get their ``__module__`` set to the target module, so
    that Sphinx ``autofunction`` and the CI stub generator attribute them to
    ``amrex.space3d`` rather than to this helper.

    Parameters
    ----------
    ns : dict
        The calling module's ``globals()``.
    amr : module
        That module's compiled bindings, e.g. ``amrex_3d_pybind``.
    """
    name = ns["__name__"]

    ns["__version__"] = amr.__version__
    ns["__doc__"] = amr.__doc__
    ns["__license__"] = amr.__license__
    ns["__author__"] = amr.__author__

    # enhance the C++ classes with methods written in pure Python
    register_Array4_extension(amr)
    register_MultiFab_extension(amr)
    register_PODVector_extension(amr)
    register_SmallMatrix_extension(amr)
    register_SoA_extension(amr)
    register_AoS_extension(amr)
    register_ParticleContainer_extension(amr)

    def Print(*args, **kwargs):
        """Wrap amrex::Print() - only the IO processor writes"""
        if not amr.initialized():
            print("warning: Print all - AMReX not initialized")
            print(*args, **kwargs)
        elif amr.ParallelDescriptor.IOProcessor():
            print(*args, **kwargs)

    def read_particles_(
        plotfile, particle_dir="particles", communicate=True, container=None
    ):
        """
        Read AMReX particle data from a plotfile/checkpoint into a container.

        See :py:func:`amrex.extensions.ParticleContainer.read_particles` for details.
        """
        return read_particles(amr, plotfile, particle_dir, communicate, container)

    read_particles_.__name__ = "read_particles"
    read_particles_.__qualname__ = "read_particles"

    class Module(types.ModuleType):
        """Module type of ``amrex.space{1,2,3}d``, adds the lazy ``xp``.

        This is used instead of a module-level PEP 562 ``__getattr__``: in the
        generated ``.pyi`` stubs, the latter tells type checkers that *any*
        attribute exists (so typos go unnoticed) and, being body-less, it
        returns ``None`` when the stubs are executed for the Sphinx docs.
        """

        @functools.cached_property
        def xp(self):
            """The array namespace matching this build.

            NumPy on CPU, CuPy for CUDA/HIP, dpnp for SYCL. It is the module
            counterpart of the ``to_xp`` methods, for code that needs to call
            into the array library itself, e.g. ``amr.xp.sin(...)``.

            Like every other CuPy/dpnp use in pyAMReX, those are optional
            dependencies: they are imported here on first access, never at
            import time, so ``import amrex`` works on a GPU build without them.
            Only touching ``amr.xp`` (or a ``to_cupy``/``to_dpnp``/``to_xp``
            call) requires one to be installed. The result is cached in the
            module ``__dict__``, so subsequent lookups are plain attribute
            lookups.

            Raises
            ------
            ImportError
                On a GPU build whose array library (CuPy or dpnp) is not
                installed.
            """
            import importlib

            from .extensions.dlpack_helpers import xp_module_name

            module_name = xp_module_name(amr)
            try:
                return importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(
                    f"amrex.xp needs {module_name!r}, which is an optional "
                    f"dependency of pyAMReX and is not installed. Install it, "
                    f"or use the to_numpy()/to_cupy()/to_dpnp() methods "
                    f"directly."
                ) from e

        def __dir__(self):
            # list xp before its first access, e.g., for tab completion
            return sorted({*super().__dir__(), "xp"})

    ns["Print"] = Print
    ns["read_particles"] = read_particles_
    ns["list_particle_species"] = list_particle_species

    for injected in (Print, read_particles_):
        injected.__module__ = name

    # a module's __class__ may be set to a ModuleType subclass, see
    # https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
    sys.modules[name].__class__ = Module
