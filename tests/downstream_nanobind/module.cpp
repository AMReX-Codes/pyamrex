#include <nanobind/nanobind.h>

#include <AMReX_IntVect.H>

namespace nb = nanobind;

amrex::IntVect const& identity (amrex::IntVect const& value)
{
    return value;
}

NB_MODULE(pyamrex_downstream, m)
{
    m.def("identity", &identity, nb::rv_policy::reference,
          "Return a borrowed pyAMReX IntVect reference.");
}
