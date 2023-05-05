/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace py = nanobind;
using namespace amrex;


void init_ParmParse(py::module_ &m) {
    py::class_<ParmParse>(m, "ParmParse")
        .def("__repr__",
             [](ParmParse const &) {
                 // todo: make ParmParse::getPrefix() public?
                 return "<amrex.ParmParse>";
             }
        )
        .def(py::init<std::string const &>(),
             py::arg("prefix") = std::string()
        )

        .def("remove", &ParmParse::remove)

        .def("addfile", &ParmParse::addfile)

        .def("add", py::overload_cast<char const*, bool const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, int const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, long const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, long long const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, float const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, double const>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, std::string const &>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, amrex::IntVect const &>(&ParmParse::add))
        .def("add", py::overload_cast<char const*, amrex::Box const &>(&ParmParse::add))

        .def("addarr", py::overload_cast<char const*, std::vector<int> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<long> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<long long> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<float> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<double> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<std::string> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<amrex::IntVect> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<char const*, std::vector<amrex::Box> const &>(&ParmParse::addarr))

        // TODO: getters and queries

        // TODO: dumpTable, hasUnusedInputs, getUnusedInputs, getEntries
    ;
}
