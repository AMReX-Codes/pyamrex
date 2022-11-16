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
        print(f"p={p}")
        if os.path.exists(p):
            os.add_dll_directory(p)
        else:
            print("  ... does NOT exist")

# import core bindings to C++
from . import amrex_pybind
from .amrex_pybind import *  # noqa

__version__ = amrex_pybind.__version__
__doc__ = amrex_pybind.__doc__
__license__ = amrex_pybind.__license__
__author__ = amrex_pybind.__author__

# at this place we can enhance Python classes with additional methods written
# in pure Python or add some other Python logic
#
