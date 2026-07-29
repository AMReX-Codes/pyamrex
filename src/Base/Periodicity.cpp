/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_Periodicity.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_SPACE.H>

#include <ios>
#include <sstream>


void init_Periodicity(nb::module_ &m)
{
    using namespace amrex;

    nb::class_< Periodicity >(m, "Periodicity")
        .def("__repr__",
            [](Periodicity const & p) {
                std::stringstream s;
                s << std::boolalpha
                  << AMREX_D_TERM(
                        p.isPeriodic(0),
                        << ", " << p.isPeriodic(1),
                        << ", " << p.isPeriodic(2));
                return "<amrex.Periodicity per direction '" + s.str() + "'>";
            }
        )

        .def(nb::init<>())
        .def(nb::init< IntVect const & >())

        .def_prop_ro("is_any_periodic", &Periodicity::isAnyPeriodic)
        .def_prop_ro("is_all_periodic", &Periodicity::isAllPeriodic)
        .def_prop_ro("domain", &Periodicity::Domain,
                               "Cell-centered domain Box \"infinitely\" long in non-periodic directions.")
        .def_prop_ro("shift_IntVect", &Periodicity::shiftIntVect)

        .def("is_periodic", &Periodicity::isPeriodic,
             nb::arg("dir"))
        .def("__getitem__", &Periodicity::isPeriodic,
             nb::arg("dir"))

        .def(nanobind::self == nanobind::self)
        //.def(nanobind::self != nanobind::self)

        .def_static("non_periodic", &Periodicity::NonPeriodic,
            "Return the Periodicity object that is not periodic in any direction")
    ;
}
