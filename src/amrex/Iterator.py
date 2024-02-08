"""
This file is part of pyAMReX

Copyright 2023 AMReX community
Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""


def next(self):
    """This is a helper function for the C++ equivalent of void operator++()

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
    if hasattr(self, "first_or_done") is False:
        self.first_or_done = True

    first_or_done = self.first_or_done
    if first_or_done:
        first_or_done = False
        self.first_or_done = first_or_done
    else:
        self._incr()
    if self.is_valid is False:
        self.first_or_done = True
        self.finalize()
        raise StopIteration

    return self
