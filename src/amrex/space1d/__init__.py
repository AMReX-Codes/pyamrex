# -*- coding: utf-8 -*-

from .._dll import add_windows_dll_directories

# Register dependent DLL locations for the C++ AMReX library and potential
# shared library dependencies before importing pybind.
add_windows_dll_directories(__file__)

# import core bindings to C++
from .._module_api import setup_module as _setup_module
from . import amrex_1d_pybind
from .amrex_1d_pybind import *  # noqa


# at this place we can enhance Python classes with additional methods written
# in pure Python or add some other Python logic
#
def d_decl(x, y, z):
    """Return a tuple of the first passed element"""
    return (x,)


# everything else is the same for every dimensionality
_setup_module(globals(), amrex_1d_pybind)
