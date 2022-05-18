/* Copyright 2022 The AMReX Community
 *
 * Authors: Ryan Sandberg
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_Particle.H>

#include <sstream>


namespace py = pybind11;
using namespace amrex;

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

// template <typename T, int NReal, int NInt>
// void make_ParticleBase(py::module &m)
// {
//     using ParticleBaseRealInt=ParticleBase<T, NReal, NInt>;
//     using ParticleBaseReal=ParticleBase<T, NReal>;
//     using ParticleBaseInt=ParticleBase<T, NInt>;
//     using ParticleBase=ParticleBase<T>;
//     py::class_<ParticleBaseRealInt>(m, "ParticleBaseRealInt")
//         .def(py::init<>())
//         .def_readwrite("m_pos")
//     ;
// }

template <int T_NReal, int T_NInt=0>
void make_Particle(py::module &m)
{
    using ParticleType = Particle<T_NReal, T_NInt>;
    auto const array_name = std::string("Particle_").append(std::to_string(T_NReal) + "_" + std::to_string(T_NInt));
    py::class_<ParticleType>(m, array_name.c_str())
        .def(py::init<>())
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto p = obj.cast<ParticleType>();
                 std::stringstream s;
                 s << p;
                 return "<amrex." + name + " " + s.str() + ">";
            }
        )
        .def("__str",
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
        .def("setPos", [](ParticleType &p, int index, Real val) { p.m_pos[index] = val; })
        .def("setPos", [](ParticleType &p, const RealVect & vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; std::cout << vals[ii] << " "; } })
        .def("setPos", [](ParticleType &p, const std::array<Real, AMREX_SPACEDIM>& vals) { for (int ii=0; ii < AMREX_SPACEDIM; ii++) { p.m_pos[ii] = vals[ii]; } })
        // .def_property_readonly
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