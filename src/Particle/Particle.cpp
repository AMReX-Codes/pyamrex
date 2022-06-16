/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

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
#include <utility>
#include <cmath>
#include <regex>



namespace py = pybind11;
using namespace amrex;
using pReal = amrex_particle_real;

struct PIdx
{
    enum RealValues { // Particle Attributes for sample particle struct-of-arrays
          w = 0,
          vx, vy, vz,
          Ex, Ey, Ez,
          nRealAttribs
    };

    enum IntValues {
        nIntAttribs
    };

};

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
    using ParticleType = Particle<T_NReal, T_NInt>;
    auto const particle_name = std::string("Particle_").append(std::to_string(T_NReal) + "_" + std::to_string(T_NInt));
    py::class_<ParticleType> (m, particle_name.c_str())
        .def(py::init<>())
        .def(py::init([](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z)) { 
                    std::unique_ptr<ParticleType> part(new ParticleType());
                    // AMREX_D_DECL(part->m_pos[0] = x;, part->m_pos[1] = y;, part->m_pos[2] = z;)
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
        .def(py::init([](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z), py::args& args) { 
                    std::unique_ptr<ParticleType> part(new ParticleType());
                    // AMREX_D_DECL(part->m_pos[0] = x;, part->m_pos[1] = y;, part->m_pos[2] = z;)
                    part->m_pos[0] = x;
    #if (AMREX_SPACEDIM >= 2)
                    part->m_pos[1] = y;
    #endif
    #if (AMREX_SPACEDIM >= 3)
                    part->m_pos[2] = z;
    #endif
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
                }
            )
        )

        .def(py::init([](AMREX_D_DECL(ParticleReal x, ParticleReal y, ParticleReal z), py::kwargs& kwargs) { 
                    std::unique_ptr<ParticleType> part(new ParticleType());
                    part->m_pos[0] = x;
    #if (AMREX_SPACEDIM >= 2)
                    part->m_pos[1] = y;
    #endif
    #if (AMREX_SPACEDIM >= 3)
                    part->m_pos[2] = z;
    #endif

                    for (auto item : kwargs) {
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
                }
            )
        )

        .def(py::init([](py::kwargs& kwargs) { 
                    std::unique_ptr<ParticleType> part(new ParticleType());
                    for (auto item : kwargs) {
                        std::regex component_separator("(.*)_([0-9]*)");
                        std::smatch sm;
                        std::string varname = item.first.cast<std::string>();
                        std::regex_match(varname, sm, component_separator, std::regex_constants::match_default);
                        int comp = -1;
                        if (varname == "x") { part->m_pos[0] = item.second.cast<ParticleReal>(); }
                        if (varname == "y") { part->m_pos[1] = item.second.cast<ParticleReal>(); }
                        if (varname == "z") { part->m_pos[2] = item.second.cast<ParticleReal>(); }
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
                }
            )
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
        // .def("cpu", py::overload_cast<>(&ParticleType::cpu))
        // .def_property_readonly("cpu", [](const ParticleType &p) { return p.cpu(); })
        // .def_property_readonly("id", [](const ParticleType &p) { return p.id(); })
        // .def("pos", py::overload_cast<int>(&ParticleType::pos, py::const_)) //, "Return specified component of particle position")
        // .def_property("pos", [](const ParticleType &p) { return p.pos(); }, [](ParticleType &p, const RealVect &rv) { p.pos = rv; })
        // .def_
        .def("pos", [](const ParticleType &p, int index) { return p.pos(index); })
        // .def("pos", py::overload_cast<>(&ParticleType::pos, py::const_))
        .def("pos", [](const ParticleType &p) { return p.pos(); })
        .def("setPos", [](ParticleType &p, int index, Real val) { AMREX_ASSERT(index > 0 && index < AMREX_SPACEDIM); p.m_pos[index] = val; })
        .def("setPos", [](ParticleType &p, const RealVect & vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; } })
        .def("setPos", [](ParticleType &p, const std::array<Real, AMREX_SPACEDIM>& vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; } })

        .def("get_rdata", [](ParticleType &p, int index) { 
                if constexpr (T_NReal > 0) {
                    if(index < 0 || index >= T_NReal) {
                        // std::string error_msg = "" 
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
        // .def("rvec")
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
        .def("set_rdata", [](ParticleType &p, const RealVect & vals) { 
                if constexpr (T_NReal > 0) {
                    for (int ii=0; ii < T_NReal; ii++) { 
                        p.m_rdata[ii] = vals[ii];  
                    }
                } else {
                    amrex::ignore_unused(vals);
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
        // template <int U = T_NInt, typename std::enable_if<U != 0, int>::type = 0>
        .def("get_idata", [](ParticleType &p, int index) { 
                if constexpr (T_NInt > 0) {
                    if(index < 0 || index >= T_NInt) {
                        // std::string error_msg = "" 
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
                        // std::string error_msg = "" 
                        throw std::range_error("index not in range. Valid range : [0, " + std::to_string(T_NInt) + ")");
                    }
                    p.m_idata[index] = val; 
                } else {
                    amrex::ignore_unused(index, val);
                }
            }
        )
        .def("set_idata", [](ParticleType &p, const IntVect & vals) { 
                if constexpr (T_NInt > 0) {
                    for (int ii=0; ii < T_NInt; ii++) { 
                        p.m_idata[ii] = vals[ii];  
                    } 
                } else {
                    amrex::ignore_unused(vals);
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
        // .def_property_readonly_static("the_next_id", [](py::object const&) {return ParticleType::the_next_id;} )
        .def("cpu", [](const ParticleType &p) { const int m_cpu = p.cpu(); return m_cpu; })
        .def("id", [](const ParticleType &p) { const int m_id = p.id(); return m_id; })
        .def("NextID", [](const ParticleType &p) {return p.NextID();})
        .def("NextID", [](const ParticleType &p, Long nextid) { p.NextID(nextid); })
        .def_property("x", [](ParticleType &p){ return p.pos(0);}, [](ParticleType &p, Real val){ p.m_pos[0] = val; })
        // if (AMREX_SPACEDIM > 1) {
#if (AMREX_SPACEDIM >= 2)
        .def_property("y", [](ParticleType &p){ return p.pos(1);}, [](ParticleType &p, Real val){ p.m_pos[1] = val; })
#endif
#if (AMREX_SPACEDIM == 3)
        .def_property("z", [](ParticleType &p){ return p.pos(2);}, [](ParticleType &p, Real val){ p.m_pos[2] = val; })
#endif
        // .def("NextID", py::overload_cast<>(&ParticleType::NextID))
        // .def("NextID", py::overload_cast<Long>(&ParticleType::NextID))
        // .def_property("rdata", [](ParticleType &p){ return p.rdata})
    ;
}



void init_Particle(py::module& m) {

    py::class_<PIdx> pidx(m, "PIdx");
    py::enum_<PIdx::RealValues>(pidx, "RealValues")
        .value("w", PIdx::RealValues::w)
        .value("vx", PIdx::RealValues::vx)
        .value("vy", PIdx::RealValues::vy)
        .value("vz", PIdx::RealValues::vz)
        .value("Ex", PIdx::RealValues::Ex)
        .value("Ey", PIdx::RealValues::Ey)
        .value("Ez", PIdx::RealValues::Ez)
    ;
    py::enum_<PIdx::IntValues>(pidx, "IntValues")
    ;
    make_Particle<PIdx::nRealAttribs, PIdx::nIntAttribs> (m);
    make_Particle<1, 1> (m);
    make_Particle<2, 1> (m);
    make_Particle<3, 2> (m);
    // make_Particle<2,2> (m);
}
