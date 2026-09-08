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


def iterate(it):
    """Drive an ``MFIter``/``ParIter`` as a generator.

    This is what ``__iter__`` returns, so it is the object a ``for`` loop holds
    on to. The ``finally`` clause runs on normal exhaustion *and* on ``break``,
    ``return`` and exceptions -- the same coverage C++ gets from ``~MFIter()``
    at scope exit.

    That matters because the C++ destructor is not a reliable stand-in here:
    ``__next__`` yields the iterator itself, so after ``break`` the loop
    variable still references it and it is not destroyed. Its ``Finalize()``
    would then be deferred, which (a) leaves ``MFIter::depth`` at 1 so the next
    iterator construction trips ``AMREX_ALWAYS_ASSERT(depth == 1)`` and aborts,
    and (b) skips the ``Gpu::streamSynchronize()`` that ``Finalize()`` performs.

    ``MFIter::Finalize()`` is idempotent (guarded by its ``finalized`` flag), so
    an explicit ``it.finalize()`` in user code stays safe.

    it: the C++ iterator to drive; yielded unchanged on every step
    """
    try:
        while it.is_valid:
            yield it
            it._incr()
    finally:
        it.finalize()


def getitem(self, name):
    """Access (read/write) particle vectors."""
    if not self.is_soa_particle:
        raise ValueError("Only pure SoA particle containers support pti.__get__")

    if name == "idcpu":
        return self.soa().get_idcpu_data().to_xp(copy=False)
    elif name in self.soa().real_names:
        return self.soa().get_real_data(name).to_xp(copy=False)
    elif name in self.soa().int_names:
        return self.soa().get_int_data(name).to_xp(copy=False)
    else:
        raise KeyError(f"Unknown particle attribute name: {name}")
