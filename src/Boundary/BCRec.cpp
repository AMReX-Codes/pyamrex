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

    // Expose AMReX boundary enums with the same names used in C++ docs.
    py::native_enum<BCType::mathematicalBndryTypes>(m, "BCType", "enum.IntEnum",
            R"pbdoc(Mathematical boundary condition types stored in BCRec.

Common values are BCType.int_dir for interior cells, BCType.foextrap
for first-order extrapolation, BCType.reflect_even and
BCType.reflect_odd for reflective boundaries, and BCType.ext_dir or
BCType.ext_dir_cc for external Dirichlet values supplied by the
application.
)pbdoc")
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
            R"pbdoc(Physical boundary condition categories.

Application code maps these physical categories to mathematical
BCType values for each field component and coordinate direction.
)pbdoc")
        .value("interior", PhysBCType::interior)
        .value("inflow", PhysBCType::inflow)
        .value("outflow", PhysBCType::outflow)
        .value("symmetry", PhysBCType::symmetry)
        .value("slipwall", PhysBCType::slipwall)
        .value("noslipwall", PhysBCType::noslipwall)
        .value("inflowoutflow", PhysBCType::inflowoutflow)
        .finalize()
    ;

    // BCRec stores low/high mathematical boundary types for one component.
    py::class_<BCRec>(m, "BCRec",
            R"pbdoc(Boundary condition record for one field component.

A BCRec stores one mathematical boundary type on the low and high side
of each coordinate direction. Pass lists of length Config.spacedim for
lo and hi, usually using BCType enum values.
)pbdoc")
        .def("__repr__",
             [](BCRec const & bcr) {
                 auto append_values = [](std::stringstream& s, int const* values) {
                     char const* separator = "";
                     for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                         s << separator << values[dir];
                         separator = ",";
                     }
                 };

                 std::stringstream s;
                 s << "<amrex.BCRec (";
                 append_values(s, bcr.lo());
                 s << ") (";
                 append_values(s, bcr.hi());
                 s << ")>";
                 return s.str();
             }
        )

        // Constructors document AMReX's component and direction conventions.
        .def(py::init<>(),
             R"pbdoc(Create a BCRec initialized to BCType.bogus on every face.

Set all low and high entries before using this record in a fill
operation.
)pbdoc")
        .def(py::init([](std::array<int, AMREX_SPACEDIM> const & lo,
                         std::array<int, AMREX_SPACEDIM> const & hi) {
                 return BCRec(lo.data(), hi.data());
             }),
             py::arg("lo"), py::arg("hi"),
             R"pbdoc(Create a BCRec from low-side and high-side boundary types.

Args:
    lo: Sequence of Config.spacedim BCType or integer values for the
        low side of each coordinate direction.
    hi: Sequence of Config.spacedim BCType or integer values for the
        high side of each coordinate direction.
)pbdoc")
        .def(py::init<Box const &, Box const &, BCRec const &>(),
             py::arg("bx"), py::arg("domain"), py::arg("bc_domain"),
             R"pbdoc(Create the BCRec for a sub-box from a domain BCRec.

For each face, the returned record inherits bc_domain when bx touches
the physical domain boundary and uses BCType.int_dir otherwise.

Args:
    bx: Box to classify.
    domain: Physical domain box.
    bc_domain: Boundary record for the full domain.
)pbdoc")

        // Mutators set one coordinate direction at a time, matching BCRec.
        .def("set_lo", &BCRec::setLo,
             py::arg("dir"), py::arg("bc_type"),
             R"pbdoc(Set the low-side boundary type in one direction.

Args:
    dir: Coordinate direction, from 0 to Config.spacedim - 1.
    bc_type: BCType or integer boundary value.
)pbdoc")
        .def("set_hi", &BCRec::setHi,
             py::arg("dir"), py::arg("bc_type"),
             R"pbdoc(Set the high-side boundary type in one direction.

Args:
    dir: Coordinate direction, from 0 to Config.spacedim - 1.
    bc_type: BCType or integer boundary value.
)pbdoc")

        // List accessors make boundary records easy to inspect from Python.
        .def("lo",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.lo(), bcr.lo() + AMREX_SPACEDIM);
             },
             "Return low-side boundary types as a list.")
        .def("lo", py::overload_cast<int>(&BCRec::lo, py::const_),
             py::arg("dir"),
             "Return the low-side boundary type in one direction.")
        .def("hi",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.hi(), bcr.hi() + AMREX_SPACEDIM);
             },
             "Return high-side boundary types as a list.")
        .def("hi", py::overload_cast<int>(&BCRec::hi, py::const_),
             py::arg("dir"),
             "Return the high-side boundary type in one direction.")
        .def("vect",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.vect(), bcr.vect() + 2*AMREX_SPACEDIM);
             },
             "Return all boundary types as low-side entries followed by high-side entries.")
        .def("data",
             [](BCRec const & bcr) {
                 return std::vector<int>(bcr.data(), bcr.data() + 2*AMREX_SPACEDIM);
             },
             "Return all boundary types as low-side entries followed by high-side entries.")

        .def(py::self == py::self)
        .def(py::self != py::self)
    ;

    make_Vector<BCRec>(m, "BCRec");

    // Return Python-owned records instead of exposing output parameters.
    m.def("setBC",
          [](Box const & bx, Box const & domain, BCRec const & bc_dom) {
              BCRec bcr;
              setBC(bx, domain, bc_dom, bcr);
              return bcr;
          },
          py::arg("bx"), py::arg("domain"), py::arg("bc_domain"),
          R"pbdoc(Return the BCRec for a box from a domain BCRec.

For each face, the returned record inherits bc_domain when bx touches
the physical domain boundary and uses BCType.int_dir otherwise.

Args:
    bx: Box to classify.
    domain: Physical domain box.
    bc_domain: Boundary record for the full domain.
)pbdoc");
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
          py::arg("bc_domain"),
          R"pbdoc(Return component boundary records for a box.

The returned Vector_BCRec has size dest_comp + ncomp. Components in
the interval [dest_comp, dest_comp + ncomp) are populated from
bc_domain[src_comp:src_comp + ncomp]. Earlier destination entries are
left at their default BCType.bogus values.

Args:
    bx: Box to classify.
    domain: Physical domain box.
    src_comp: First component to read from bc_domain.
    dest_comp: First component to write in the returned Vector_BCRec.
    ncomp: Number of component records to populate.
    bc_domain: Domain boundary records for source components.
)pbdoc");
}
