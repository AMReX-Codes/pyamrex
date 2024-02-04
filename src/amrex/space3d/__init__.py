# -*- coding: utf-8 -*-

import os

# Python 3.8+ on Windows: DLL search paths for dependent
# shared libraries
# Refs.:
# - https://github.com/python/cpython/issues/80266
# - https://docs.python.org/3.8/library/os.html#os.add_dll_directory
if os.name == "nt":
    # add anything in the current directory
    pwd = __file__.rsplit(os.sep, 1)[0] + os.sep
    os.add_dll_directory(pwd)
    # add anything in PATH
    paths = os.environ.get("PATH", "")
    for p in paths.split(";"):
        if os.path.exists(p):
            os.add_dll_directory(p)

# import core bindings to C++
from . import amrex_3d_pybind
from .amrex_3d_pybind import *  # noqa

__version__ = amrex_3d_pybind.__version__
__doc__ = amrex_3d_pybind.__doc__
__license__ = amrex_3d_pybind.__license__
__author__ = amrex_3d_pybind.__author__


# at this place we can enhance Python classes with additional methods written
# in pure Python or add some other Python logic
#
def d_decl(x, y, z):
    """Return a tuple of the three passed elements"""
    return (x, y, z)


def Print(*args, **kwargs):
    """Wrap amrex::Print() - only the IO processor writes"""
    if not initialized():  # noqa
        print("warning: Print all - AMReX not initialized")
        print(*args, **kwargs)
    elif ParallelDescriptor.IOProcessor():  # noqa
        print(*args, **kwargs)


from ..Array4 import register_Array4_extension
from ..ArrayOfStructs import register_AoS_extension
from ..MultiFab import register_MultiFab_extension
from ..PODVector import register_PODVector_extension
from ..ParticleComponentNames import soa_int_comps, soa_real_comps  # noqa
from ..ParticleContainer import register_ParticleContainer_extension
from ..StructOfArrays import register_SoA_extension

register_Array4_extension(amrex_3d_pybind)
register_MultiFab_extension(amrex_3d_pybind)
register_PODVector_extension(amrex_3d_pybind)
register_SoA_extension(amrex_3d_pybind)
register_AoS_extension(amrex_3d_pybind)
register_ParticleContainer_extension(amrex_3d_pybind)
