import amrex.space3d as amr
import pyamrex_downstream


value = amr.IntVect(1, 2, 3)
result = pyamrex_downstream.identity(value)

assert result is value
assert tuple(result) == (1, 2, 3)
