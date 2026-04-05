"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def podvector_to_numpy(self, copy=False):
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
    import numpy as np

    if self.size() > 0:
        if copy:
            # This supports a device-to-host copy.
            #
            # The to_host() returned object is a temporary, and
            # np.array using the __array_interface__ protocol does
            # not keep it alive automatically unless it is stored
            # in an actual variable (tmp).
            tmp = self.to_host()
            ret = np.array(tmp, copy=False)
            assert ret.base is tmp
            return ret
        else:
            return np.array(self, copy=False)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_cupy(self, copy=False):
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
    import cupy as cp

    if self.size() > 0:
        return cp.array(self, copy=copy)
    else:
        raise ValueError("Vector is empty.")


def podvector_to_xp(self, copy=False):
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
    import inspect

    amr = inspect.getmodule(self)
    return self.to_cupy(copy) if amr.Config.have_gpu else self.to_numpy(copy)


def _is_host_accessible(cls):
    """Check if a PODVector type's allocator provides host-accessible memory.

    On CPU builds all allocators are host-accessible.  On GPU builds, AMReX
    arenas decide this at runtime; this matters for the default and
    polymorphic arenas.
    """
    import inspect

    amr = inspect.getmodule(cls)
    if not amr.Config.have_gpu:
        return True

    suffix = cls.__name__.rsplit("_", 1)[-1]
    if suffix == "std":
        return True

    arenas = {
        "arena": amr.The_Arena,
        "device": amr.The_Device_Arena,
        "pinned": amr.The_Pinned_Arena,
        "managed": amr.The_Managed_Arena,
        "async": amr.The_Async_Arena,
        "polymorphic": amr.The_Arena,
    }
    return arenas[suffix]().is_host_accessible


def podvector_from_numpy(cls, arr):
    """
    Create a new PODVector from a NumPy array (or array-like).

    Always copies the data into a newly allocated PODVector.
    For device-only allocators, the input is staged through CuPy.

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data, convertible to a NumPy array.

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.

    """
    import numpy as np

    arr_np = np.asarray(arr)
    n = len(arr_np)
    if n == 0:
        return cls()

    pv = cls(n)
    if _is_host_accessible(cls):
        np.array(pv, copy=False)[:] = arr_np
    else:
        import cupy as cp

        cp.asarray(pv)[:] = cp.asarray(arr_np)
    return pv


def podvector_from_cupy(cls, arr):
    """
    Create a new PODVector from a CuPy array (or array-like).

    Always copies the data into a newly allocated PODVector.
    Works for every allocator type: for host-only allocators the
    data is staged to the host through NumPy automatically.

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data, convertible to a CuPy array.

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.
    """
    import cupy as cp

    arr_cp = cp.asarray(arr)
    n = len(arr_cp)
    if n == 0:
        return cls()
    pv = cls(n)
    if _is_host_accessible(cls):
        import numpy as np

        np.array(pv, copy=False)[:] = cp.asnumpy(arr_cp)
    else:
        cp.asarray(pv)[:] = arr_cp
    return pv


def podvector_from_xp(cls, arr):
    """
    Create a new PODVector from a NumPy or CuPy array,
    depending on amr.Config.have_gpu .

    Always copies the data into a newly allocated PODVector.
    Unlike :meth:`to_xp`, a zero-copy view is not possible here because
    PODVector always owns its memory through its allocator.

    This function is similar to CuPy's xp naming suggestion for CPU/GPU agnostic code:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code

    Parameters
    ----------
    cls : type
        The PODVector type to construct.
    arr : array_like
        Input data (NumPy or CuPy array).

    Returns
    -------
    PODVector
        A new PODVector with a copy of the data.
    """
    if _is_host_accessible(cls):
        return cls.from_numpy(arr)
    else:
        return cls.from_cupy(arr)


def register_PODVector_extension(amr):
    """PODVector helper methods"""
    import inspect
    import sys

    # register member functions for every PODVector_* type
    for _, POD_type in inspect.getmembers(
        sys.modules[amr.__name__],
        lambda member: (
            inspect.isclass(member)
            and member.__module__ == amr.__name__
            and member.__name__.startswith("PODVector_")
        ),
    ):
        # instance methods: PODVector -> array
        POD_type.to_numpy = podvector_to_numpy
        POD_type.to_cupy = podvector_to_cupy
        POD_type.to_xp = podvector_to_xp

        # class methods: array -> PODVector
        POD_type.from_numpy = classmethod(podvector_from_numpy)
        POD_type.from_cupy = classmethod(podvector_from_cupy)
        POD_type.from_xp = classmethod(podvector_from_xp)
