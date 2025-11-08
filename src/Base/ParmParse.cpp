/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>

#include <functional>
#include <iostream>
#include <string>
#include <string_view>
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

        .def("add", [](ParmParse &pp, std::string_view name, bool val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, int val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, long val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, long long val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, float val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, double val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, std::string const &val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, amrex::IntVect const &val) { pp.add(name, val); })
        .def("add", [](ParmParse &pp, std::string_view name, amrex::Box const &val) { pp.add(name, val); })
        .def("addarr", py::overload_cast<std::string_view, std::vector<int> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<long> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<long long> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<float> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<double> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<std::string> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<amrex::IntVect> const &>(&ParmParse::addarr))
        .def("addarr", py::overload_cast<std::string_view, std::vector<amrex::Box> const &>(&ParmParse::addarr))

        // TODO: getters and queries
        .def("get_bool",
            [](ParmParse &pp, std::string name, int ival) {
                bool ref;
                pp.get(name, ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
         )

        .def("get_int",
            [](ParmParse &pp, std::string name, int ival) {
                int ref;
                pp.get(name, ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
         )

        .def("get_real",
            [](ParmParse &pp, std::string name, int ival) {
                amrex::Real ref;
                pp.get(name, ref, ival);
                return ref;
            },
            "parses input values", py::arg("name"), py::arg("ival")=0
        )

        .def("query_int",
            [](ParmParse &pp, std::string name, int ival) {
                int ref;
                bool exist = pp.query(name, ref, ival);
                return std::make_tuple(exist,ref);
            },
            "queries input values", py::arg("name"), py::arg("ival")=0
        )

        .def(
            "pretty_print_table",
            [](ParmParse &pp) {
                py::scoped_ostream_redirect stream(
                    std::cout,                               // std::ostream&
                    py::module_::import("sys").attr("stdout") // Python output
                );
                pp.prettyPrintTable(std::cout);
            },
            "Write the table in a pretty way to the ostream. If there are "
            "duplicates, only the last one is printed."
        )

        // TODO: dumpTable, hasUnusedInputs, getUnusedInputs, getEntries

        .def(
            "to_dict",
            [](ParmParse &pp) {
                py::dict d;

                auto g_table = pp.table();

                // sort all keys
                std::vector<std::string> sorted_names;
                sorted_names.reserve(g_table.size());
                for (auto const& [name, entry] : g_table) {
                    sorted_names.push_back(name);
                }
                std::sort(sorted_names.begin(), sorted_names.end());

                // helper function to unroll any nested parameter.sub.options into dict of dict of value
                auto add_nested = [&d](auto && value, std::string_view s) {
                    py::dict d_inner = d;  // just hold a handle to the current dict

                    while (true) {
                        auto pos = s.find('.');
                        bool last = pos == std::string_view::npos;
                        py::str key(s.substr(0, pos));

                        if (last) {
                            d_inner[key] = value;
                            break;
                        } else {
                            // Create nested dict if missing or wrong type
                            if (!d_inner.contains(key) ||
                                !py::isinstance<py::dict>(d_inner[key])) {
                                d_inner[key] = py::dict();
                            }

                            // Move one level deeper (safe, keeps reference alive)
                            d_inner = d_inner[key].cast<py::dict>();
                        }
                        s.remove_prefix(pos + 1);
                    }
                };

                for (auto const& name : sorted_names) {
                    auto const& entry = g_table[name];
                    for (auto const & vals : entry.m_vals) {
                        if (vals.size() == 1) {
                            std::visit(
                                [&](auto&& arg) {
                                    using T = std::remove_pointer_t<std::decay_t<decltype(arg)>>;
                                    T v;
                                    pp.get(name, v);
                                    add_nested(v, name);
                                },
                                entry.m_typehint
                            );
                        } else {
                            std::visit(
                                [&](auto&& arg) {
                                    using T = std::remove_pointer_t<std::decay_t<decltype(arg)>>;
                                    if constexpr (!std::is_same_v<T, bool>) {
                                        std::vector<T> valarr;
                                        pp.getarr(name, valarr);
                                        add_nested(valarr, name);
                                    }
                                },
                                entry.m_typehint
                            );
                        }
                    }
                }

                return d;
            },
            R"(Convert to a nested Python dictionary.

.. code-block:: python

    # Example: dump all ParmParse entries to YAML or TOML
    import toml
    import yaml

    pp = amr.ParmParse("").to_dict()
    yaml_string = yaml.dump(d)
    toml_string = toml.dumps(d)
)"
        )
    ;
}
