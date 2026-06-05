/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_Vector.H>

#include <array>
#include <sstream>
#include <vector>


void init_BCRec(py::module& m)
{
    using namespace amrex;

    py::native_enum<BCType::mathematicalBndryTypes>(m, "BCType", "enum.IntEnum",
            "Mathematical boundary condition types")
        .value("bogus", BCType::bogus)
        .value("reflect_odd", BCType::reflect_odd)
        .value("int_dir", BCType::int_dir)
        .value("reflect_even", BCType::reflect_even)
        .value("foextrap", BCType::foextrap)
        .value("ext_dir", BCType::ext_dir)
        .value("hoextrap", BCType::hoextrap)
        .value("hoextrapcc", BCType::hoextrapcc)
        .value("ext_dir_cc", BCType::ext_dir_cc)
        .value("direction_dependent", BCType::direction_dependent)
        .value("user_1", BCType::user_1)
        .value("user_2", BCType::user_2)
        .value("user_3", BCType::user_3)
        .finalize()
    ;

    py::native_enum<PhysBCType::physicalBndryTypes>(m, "PhysBCType", "enum.IntEnum",
            "Physical boundary condition types")
        .value("interior", PhysBCType::interior)
        .value("inflow", PhysBCType::inflow)
        .value("outflow", PhysBCType::outflow)
        .value("symmetry", PhysBCType::symmetry)
        .value("slipwall", PhysBCType::slipwall)
        .value("noslipwall", PhysBCType::noslipwall)
        .value("inflowoutflow", PhysBCType::inflowoutflow)
        .finalize()
    ;

    py::class_<BCRec>(m, "BCRec",
            "Boundary Condition Records. Necessary information and "
            "functions for computing boundary conditions.")
        .def("__repr__",
             [](BCRec const & bcr) {
                 std::stringstream s;
                 s << "<amrex.BCRec (";
                 for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                     s << bcr.lo(dir);
                     s << (dir < AMREX_SPACEDIM-1 ? "," : ") (");
                 }
                 for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                     s << bcr.hi(dir);
                     s << (dir < AMREX_SPACEDIM-1 ? "," : ")>");
                 }
                 return s.str();
             }
        )

        .def(py::init<>(),
             "The default constructor, which does NOT set valid boundary types.")
        .def(py::init([](std::array<int, AMREX_SPACEDIM> const & lo,
                         std::array<int, AMREX_SPACEDIM> const & hi) {
                 return BCRec(lo.data(), hi.data());
             }),
             py::arg("lo"), py::arg("hi"),
             "The constructor, taking the boundary condition types on the "
             "low and high side of the domain, per direction.")
        .def(py::init<Box const &, Box const &, BCRec const &>(),
             py::arg("bx"), py::arg("domain"), py::arg("bc_domain"),
             "Yet another constructor. Inherits bndry types from bc_domain "
             "when bx lies on edge of domain otherwise gets interior Dirichlet.")

        .def("set_lo", &BCRec::setLo,
             py::arg("dir"), py::arg("bc_val"),
             "Explicitly set lo bndry value.")
        .def("set_hi", &BCRec::setHi,
             py::arg("dir"), py::arg("bc_val"),
             "Explicitly set hi bndry value.")

        .def("lo",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.lo(), bcr.lo() + AMREX_SPACEDIM);
             },
             "Return low-end boundary data.")
        .def("lo", py::overload_cast<int>(&BCRec::lo, py::const_),
             py::arg("dir"),
             "Return low-end boundary data in direction dir.")
        .def("hi",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.hi(), bcr.hi() + AMREX_SPACEDIM);
             },
             "Return high-end boundary data.")
        .def("hi", py::overload_cast<int>(&BCRec::hi, py::const_),
             py::arg("dir"),
             "Return high-end boundary data in direction dir.")
        .def("vect",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.vect(), bcr.vect() + 2*AMREX_SPACEDIM);
             },
             "Return bndry values.")
        .def("data",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.data(), bcr.data() + 2*AMREX_SPACEDIM);
             },
             "Return bndry values.")

        .def(py::self == py::self)
        .def(py::self != py::self)
    ;

    make_Vector<BCRec>(m, "BCRec");

    m.def("setBC",
          [](Box const & bx, Box const & domain, BCRec const & bc_dom) {
              BCRec bcr;
              setBC(bx, domain, bc_dom, bcr);
              return bcr;
          },
          py::arg("bx"), py::arg("domain"), py::arg("bc_dom"),
          "Function for setting a BC. Inherits bndry types from bc_dom "
          "when bx lies on edge of domain otherwise gets interior Dirichlet.");
    m.def("setBC",
          [](Box const & bx, Box const & domain,
             int src_comp, int dest_comp, int ncomp,
             Vector<BCRec> const & bc_dom) {
              Vector<BCRec> bcr(dest_comp + ncomp);
              setBC(bx, domain, src_comp, dest_comp, ncomp, bc_dom, bcr);
              return bcr;
          },
          py::arg("bx"), py::arg("domain"),
          py::arg("src_comp"), py::arg("dest_comp"), py::arg("ncomp"),
          py::arg("bc_dom"),
          "Function for setting array of BCs.");
}
