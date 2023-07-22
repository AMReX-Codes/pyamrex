/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>

#include <string>
#include <vector>


void init_ParmParse(py::module &m)
{
    using namespace amrex;

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

        .def_static("addfile", &ParmParse::addfile)

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
        .def("get_bool",
            [](ParmParse &pp, std::string name, int ival) {
                bool ref;
                pp.get(name.c_str(), ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
         )

        .def("get_int",
            [](ParmParse &pp, std::string name, int ival) {
                int ref;
                pp.get(name.c_str(), ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
         )

        .def("get_real",
            [](ParmParse &pp, std::string name, int ival) {
                amrex::Real ref;
                pp.get(name.c_str(), ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
         )

         .def("query_int",
             [](ParmParse &pp, std::string name, int ival) {
                 int ref;
                 bool exist = pp.query(name.c_str(), ref, ival);
                 return std::make_tuple(exist,ref);
             },
             "queries input values", py::arg("name"), py::arg("ival")=0
         )

        // TODO: dumpTable, hasUnusedInputs, getUnusedInputs, getEntries
    ;
}
