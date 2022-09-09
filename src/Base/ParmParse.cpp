/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>

#include <string>
#include <vector>

namespace py = pybind11;
using namespace amrex;


void init_ParmParse(py::module &m) {
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
