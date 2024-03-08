/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_Particle.H>

#include <array>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <regex>


namespace
{
    /** Build a std::array from a fixed-size C array at compile-time */
    template<typename T, std::size_t... I>
    constexpr auto
    build_array (T a[], std::index_sequence<I...> s)
    {
        return std::array<T, s.size()>{a[I]...};
    }
}

template <int T_NReal, int T_NInt=0>
void make_Particle(py::module &m)
{
    // For legacy AoS + SoA particles, we register a particle and superparticle.
    // The latter adds the SoA attributes. Avoid to double-register those types.
    /*
    static std::unordered_set<std::array<int, 2>> registered;
    std::array<int, 2> const this_p = {T_NReal, T_NInt}
    if (auto search = registered.find(this_p); search != registered.end()) {
        return;
    }
    registered.insert(this_p);
    */

    using namespace amrex;

    using ParticleType = Particle<T_NReal, T_NInt>;
    auto const particle_name = std::string("Particle_").append(std::to_string(T_NReal) + "_" + std::to_string(T_NInt));
    py::class_<ParticleType> (m, particle_name.c_str())
        .def(py::init<>())
        .def(py::init([](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z)) {
                    std::unique_ptr<ParticleType> part(new ParticleType());
                    part->m_pos[0] = x;
    #if (AMREX_SPACEDIM >= 2)
                    part->m_pos[1] = y;
    #endif
    #if (AMREX_SPACEDIM >= 3)
                    part->m_pos[2] = z;
    #endif
                    return part;
                }
            )
        )
        .def(py::init(
            [](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z), py::args& args)
            {
                std::unique_ptr<ParticleType> part(new ParticleType());
                AMREX_D_TERM(part->m_pos[0] = x;, part->m_pos[1] = y;, part->m_pos[2] = z;)
                int T_NTotal = T_NReal + T_NInt;
                int argn = args.size();
                if(argn != T_NTotal) {
                    throw std::runtime_error("Must supply all " + std::to_string(T_NTotal) + " rdata, idata elements");
                }
                if constexpr (T_NReal > 0) {
                    for (int ii = 0; ii < T_NReal; ii++) {
                        part->m_rdata[ii] = py::cast<ParticleReal>(args[ii]);
                    }
                }
                if constexpr (T_NInt > 0) {
                    for (int ii = 0; ii < T_NInt; ii++) {
                        part->m_idata[ii] = py::cast<int>(args[ii+T_NReal]);
                    }
                }
                return part;
            })
        )

        .def(py::init(
            [](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z), py::kwargs& kwargs)
            {
                std::unique_ptr<ParticleType> part(new ParticleType());
                AMREX_D_TERM(part->m_pos[0] = x;, part->m_pos[1] = y;, part->m_pos[2] = z;)

                for (auto const& item : kwargs) {
                    std::regex component_separator("(.*)_([0-9]*)");
                    std::smatch sm;
                    std::string varname = item.first.cast<std::string>();
                    std::regex_match(varname, sm, component_separator, std::regex_constants::match_default);
                    int comp = std::stoi(sm[2]);
                    if constexpr (T_NReal > 0) {
                        if (comp >= 0 && comp < T_NReal && sm[1] == "rdata") {
                            part->m_rdata[comp] = item.second.cast<ParticleReal>();
                        }
                    }
                    if constexpr (T_NInt > 0) {
                        if (comp >= 0 && comp < T_NInt && sm[1] == "idata") {
                            part->m_idata[comp] = item.second.cast<int>();
                        }
                    }
                }
                return part;
            })
        )

        .def(py::init(
            [](py::kwargs& kwargs) {
                std::unique_ptr<ParticleType> part(new ParticleType());
                for (auto const& item : kwargs) {
                    std::regex component_separator("(.*)_([0-9]*)");
                    std::smatch sm;
                    std::string varname = item.first.cast<std::string>();
                    std::regex_match(varname, sm, component_separator, std::regex_constants::match_default);
                    int comp = -1;
                    if (varname == "x") { part->m_pos[0] = item.second.cast<ParticleReal>(); }
#if AMREX_SPACEDIM >= 2
                    if (varname == "y") { part->m_pos[1] = item.second.cast<ParticleReal>(); }
#elif AMREX_SPACEDIM == 3
                    if (varname == "z") { part->m_pos[2] = item.second.cast<ParticleReal>(); }
#endif
                    if (sm.size() > 2) {
                        comp = std::stoi(sm[2]);
                        if constexpr (T_NReal > 0) {
                            if(comp >= 0 && comp < T_NReal && sm[1] == "rdata") {
                                part->m_rdata[comp] = item.second.cast<ParticleReal>();
                            }
                        }
                        if constexpr (T_NInt > 0) {
                            if (comp >= 0 && comp < T_NInt && sm[1] == "idata") {
                                part->m_idata[comp] = item.second.cast<int>();
                            }
                        }
                    }
                }
                return part;
            })
        )
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto p = obj.cast<ParticleType>();
                 std::stringstream s;
                 s << p;
                 return "<amrex." + name + " with attributes\nid cpu pos rdata idata \n" + s.str() + ">";
            }
        )
        .def("__str__",
             [](const ParticleType& p) {
                 std::stringstream s;
                 s << p;
                 return s.str();
             })
        .def_readonly_static("NReal", &ParticleType::NReal)
        .def_readonly_static("NInt", &ParticleType::NInt)
        .def("pos", [](const ParticleType &p, int index) { return p.pos(index); })
        .def("pos", [](const ParticleType &p) { return p.pos(); })
        .def("setPos", [](ParticleType &p, int index, Real val) { AMREX_ASSERT(index > 0 && index < AMREX_SPACEDIM); p.m_pos[index] = val; })
        .def("setPos", [](ParticleType &p, const RealVect & vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; } })
        .def("setPos", [](ParticleType &p, const std::array<Real, AMREX_SPACEDIM>& vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; } })

        .def("get_rdata", [](ParticleType &p, int index) {
                if constexpr (T_NReal > 0) {
                    if(index < 0 || index >= T_NReal) {
                        throw std::range_error("index not in range. Valid range : [0, " + std::to_string(T_NReal));
                    }
                    return p.m_rdata[index];
                } else {
                    amrex::ignore_unused(p, index);
                    return py::none();
                }
            }
        )
        .def("get_rdata", [](ParticleType &p) {
                if constexpr (T_NReal > 0) {
                    return build_array(
                        p.m_rdata,
                        std::make_index_sequence<T_NReal>{}
                    );
                } else {
                    amrex::ignore_unused(p);
                    return py::none();
                }
            }
        )
        .def("set_rdata", [](ParticleType &p, int index, Real val) {
                if constexpr (T_NReal > 0) {
                    if(index < 0 || index >= T_NReal) {
                        // std::string error_msg = ""
                        throw std::range_error("index not in range. Valid range : [0, " + std::to_string(T_NReal) + ")");
                    }
                    p.m_rdata[index] = val;
                } else {
                    amrex::ignore_unused(index, val);
                }
            }
        )
        .def("set_rdata", [](ParticleType &p, const std::array<Real, T_NReal>& vals) {
                if constexpr (T_NReal > 0) {
                    for (int ii=0; ii < T_NReal; ii++) {
                        p.m_rdata[ii] = vals[ii];
                    }
                } else {
                    amrex::ignore_unused(p, vals);
                }
            }
        )
        .def("get_idata", [](ParticleType &p, int index) {
                if constexpr (T_NInt > 0) {
                    if(index < 0 || index >= T_NInt) {
                        throw std::range_error("index not in range. Valid range : [0, " + std::to_string(T_NInt));
                    }
                    return p.m_idata[index];
                } else {
                    amrex::ignore_unused(p, index);
                    return py::none();
                }
            }
        )
        .def("get_idata", [](ParticleType &p) {
                if constexpr (T_NInt > 0) {
                    return build_array(
                        p.m_idata,
                        std::make_index_sequence<T_NInt>{}
                    );
                }
                else {
                    amrex::ignore_unused(p);
                    return py::none();
                }
            }
        )
        .def("set_idata", [](ParticleType &p, int index, int val) {
                if constexpr (T_NInt > 0) {
                    if(index < 0 || index >= T_NInt) {
                        throw std::range_error("index not in range. Valid range : [0, " + std::to_string(T_NInt) + ")");
                    }
                    p.m_idata[index] = val;
                } else {
                    amrex::ignore_unused(index, val);
                }
            }
        )
        .def("set_idata", [](ParticleType &p, const std::array<int, T_NInt>& vals) {
                if constexpr (T_NInt > 0) {
                    for (int ii=0; ii < T_NInt; ii++) {
                        p.m_idata[ii] = vals[ii];
                    }
                } else {
                    amrex::ignore_unused(vals);
                }
            }
        )
        .def("cpu", [](const ParticleType &p) { const int m_cpu = p.cpu(); return m_cpu; })
        .def("id", [](const ParticleType &p) { const int m_id = p.id(); return m_id; })
        .def("NextID", [](const ParticleType &p) {return p.NextID();})
        .def("NextID", [](const ParticleType &p, Long nextid) { p.NextID(nextid); })
        .def_property("x", [](ParticleType &p){ return p.pos(0);}, [](ParticleType &p, Real val){ p.m_pos[0] = val; })
#if AMREX_SPACEDIM >= 2
        .def_property("y", [](ParticleType &p){ return p.pos(1);}, [](ParticleType &p, Real val){ p.m_pos[1] = val; })
#endif
#if AMREX_SPACEDIM == 3
        .def_property("z", [](ParticleType &p){ return p.pos(2);}, [](ParticleType &p, Real val){ p.m_pos[2] = val; })
#endif
    ;
}
