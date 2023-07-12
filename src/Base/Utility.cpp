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
          [] (const std::string& root,
                     int                num,
                     int                mindigits) {
          return amrex::Concatenate(root, num, mindigits);
          }, py::return_value_policy::move,
          "Builds plotfile name");
}
