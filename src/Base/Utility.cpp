/* Copyright 2021-2022 The AMReX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Utility.H>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;
using namespace amrex;

void init_Utility(py::module& m)
{
    m.def("concatenate",
          &amrex::Concatenate,
          "Builds plotfile name",
          py::arg("root"), py::arg("num"), py::arg("mindigits")=5
    );
}
