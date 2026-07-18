#include "pyAMReX.H"

#include "Base/Vector.H"

#include <AMReX_Geometry.H>
#include <AMReX_CoordSys.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>
#include <AMReX_REAL.H>

#include <sstream>
#include <string>
#include <stdexcept>


void init_Geometry(nb::module_& m)
{
    using namespace amrex;

    nb::class_<GeometryData>(m, "GeometryData")
        .def("__repr__",
            [](const GeometryData&) {
                return "<amrex.GeometryData>";
            }
        )
        .def(nb::init<>())
        .def_ro("prob_domain", &GeometryData::prob_domain, "The problem domain (real).")
        .def_ro("domain", &GeometryData::domain, "The index domain.")
        .def_ro("coord", &GeometryData::coord, "The Coordinates type.")
        .def_prop_ro("dx",
            [](const GeometryData& gd){
                std::array<Real,AMREX_SPACEDIM> dx {AMREX_D_DECL(
                    gd.dx[0], gd.dx[1], gd.dx[2]
                )};
                return dx;
            },
            "The cellsize for each coordinate direction."
        )
        .def_prop_ro("is_periodic",
            [](const GeometryData& gd){
                std::array<int,AMREX_SPACEDIM> per {AMREX_D_DECL(
                    gd.is_periodic[0], gd.is_periodic[1], gd.is_periodic[2]
                )};
                return per;
            },
            "Returns whether the domain is periodic in each coordinate direction."
        )
            //     ,
            // [](GeometryData& gd, std::vector<Real> per_in) {
            //     AMREX_D_TERM(gd.is_periodic[0] = per_in[0];,
            //                  gd.is_periodic[1] = per_in[1];,
            //                  gd.is_periodic[2] = per_in[2];)
            // })

        .def("CellSize", [](const GeometryData& gd) {
                std::array<Real,AMREX_SPACEDIM> cell_size {AMREX_D_DECL(
                    gd.CellSize(0), gd.CellSize(1), gd.CellSize(2)
                )};
                return cell_size;},
            "Returns the cellsize for each coordinate direction.")
        .def("CellSize", [](const GeometryData& gd, int comp) { return gd.CellSize(comp);},
            "Returns the cellsize for specified coordinate direction.")
        .def("ProbLo", [](const GeometryData& gd) {
                std::array<Real,AMREX_SPACEDIM> lo {AMREX_D_DECL(
                    gd.ProbLo(0), gd.ProbLo(1), gd.ProbLo(2)
                )};
                return lo;},
            "Returns the lo end for each coordinate direction.")
        .def("ProbLo", [](const GeometryData& gd, int comp) { return gd.ProbLo(comp);},
            "Returns the lo end of the problem domain in specified dimension.")
        .def("ProbHi", [](const GeometryData& gd) {
                std::array<Real,AMREX_SPACEDIM> hi {AMREX_D_DECL(
                    gd.ProbHi(0), gd.ProbHi(1), gd.ProbHi(2)
                )};
                return hi;},
            "Returns the hi end for each coordinate direction.")
        .def("ProbHi", [](const GeometryData& gd, int comp) { return gd.ProbHi(comp);},
            "Returns the hi end of the problem domain in specified dimension.")
        .def("Domain", &GeometryData::Domain,
            "Returns our rectangular domain")
        .def("isPeriodic", [](const GeometryData& gd) {
                std::array<int,AMREX_SPACEDIM> per {AMREX_D_DECL(
                    gd.isPeriodic(0), gd.isPeriodic(1), gd.isPeriodic(2)
                )};
                return per;},
            "Returns whether the domain is periodic in each direction.")
        .def("isPeriodic", &GeometryData::isPeriodic,
            "Returns whether the domain is periodic in the given direction.")
        .def("Coord", &GeometryData::Coord,"return integer coordinate type")
    ;

    nb::class_<Geometry, CoordSys>(m, "Geometry")
        .def("__repr__",
             [](nb::object& obj) {
                 nb::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = nb::cast<std::string>(py_name);
                 const auto gm = nb::cast<Geometry>(obj);
                 std::stringstream s;
                 s << gm;
                 return "<amrex." + name + " " + s.str() + ">";
            }
        )
        .def("__str__",
             [](const Geometry& gm) {
                 std::stringstream s;
                 s << gm;
                 return s.str();
             })
        .def(nb::init<>())
        .def(nb::init<
            const Box&,
            const RealBox&,
            int,
            Array<int, AMREX_SPACEDIM> const&
          >(),
          nb::arg("dom"), nb::arg("rb"), nb::arg("coord"), nb::arg("is_per"))

        .def("data", &Geometry::data, "Returns non-static copy of geometry's stored data")
        // .def("setup")

        .def("ResetDefaultProbDomain",
            nb::overload_cast<const RealBox&>
            (&Geometry::ResetDefaultProbDomain),
            "Reset default problem domain of Geometry class with a `RealBox`")
        .def("ResetDefaultPeriodicity",
            nb::overload_cast<const Array<int,AMREX_SPACEDIM>& >
            (&Geometry::ResetDefaultPeriodicity),
            "Reset default periodicity of Geometry class with an Array of `int`")
        .def("ResetDefaultCoord",
            nb::overload_cast< int >
            (&Geometry::ResetDefaultCoord),
            "Reset default coord of Geometry class with an Array of `int`")

        .def("define", nb::overload_cast<const Box&, const RealBox&,
                                        int, Array<int,AMREX_SPACEDIM> const&>
                                        (&Geometry::define),
            nb::arg("dom"), nb::arg("rb"), nb::arg("coord"), nb::arg("is_per"),
            "Set geometry"
        )

        .def_prop_rw("prob_domain",
            nb::overload_cast<>(&Geometry::ProbDomain, nb::const_),
            nb::overload_cast<RealBox const &>(&Geometry::ProbDomain),
            "The problem domain (real)."
        )
        .def("ProbLo", nb::overload_cast<int>(&Geometry::ProbLo, nb::const_),
            nb::arg("dir"),
            "Get the lo end of the problem domain in specified direction")
        .def("ProbLo",
            [](const Geometry& gm) {
                Array<Real,AMREX_SPACEDIM> lo {{AMREX_D_DECL(gm.ProbLo(0),gm.ProbLo(1),gm.ProbLo(2))}};
                return lo;
            },
            "Get the list of lo ends of the problem domain"
        )
        .def("ProbHi", nb::overload_cast<int>(&Geometry::ProbHi, nb::const_),
             nb::arg("dir"),
            "Get the hi end of the problem domain in specified direction")
        .def("ProbHi",
            [](const Geometry& gm) {
                Array<Real,AMREX_SPACEDIM> hi {{AMREX_D_DECL(gm.ProbHi(0),gm.ProbHi(1),gm.ProbHi(2))}};
                return hi;
            },
            "Get the list of lo ends of the problem domain"
        )
        .def("ProbSize", &Geometry::ProbSize, "the overall size of the domain")
        .def("ProbLength", &Geometry::ProbLength, "length of problem domain in specified dimension")

        .def_prop_rw("domain",
              nb::overload_cast<>(&Geometry::Domain, nb::const_),
              nb::overload_cast<Box const &>(&Geometry::Domain),
              "The rectangular domain (index space)."
        )

        // GetVolume
        // .def("GetVolume", nb::overload_cast<MultiFab&>(&Geometry::GetVolume, nb::const_))
        // .def("GetVolume", nb::overload_cast<)
        // ---- needs FArrayBox, BoxArray ! --------
        // GetDLogA
        // GetFaceArea

        .def("isPeriodic", nb::overload_cast<int>(&Geometry::isPeriodic, nb::const_),
            "Is the domain periodic in the specified direction?")
        .def("isAnyPeriodic", nb::overload_cast<>(&Geometry::isAnyPeriodic, nb::const_),
            "Is domain periodic in any direction?")
        .def("isAllPeriodic", nb::overload_cast<>(&Geometry::isAllPeriodic, nb::const_),
            "Is domain periodic in all directions?")
        .def("isPeriodic", nb::overload_cast<>(&Geometry::isPeriodic, nb::const_),
            "Return list indicating whether domain is periodic in each direction")
        .def("period",
            [](const Geometry& gm, const int dir) {
                if(gm.isPeriodic(dir)){ return gm.period(dir); }
                else { throw std::runtime_error("Geometry is not periodic in this direction."); }
            },
            nb::arg("dir"),
            "Return the period in the specified direction")
        .def("periodicity",
            nb::overload_cast<>(&Geometry::periodicity, nb::const_)
        )
        .def("periodicity",
            nb::overload_cast<const Box&>(&Geometry::periodicity, nb::const_),
            nb::arg("b"),
            "Return Periodicity object with lengths determined by input Box"
        )

        // .def("periodicShift", &Geometry::periodicShift)
        .def("growNonPeriodicDomain", nb::overload_cast<IntVect const&>(&Geometry::growNonPeriodicDomain, nb::const_),
            nb::arg("ngrow"))
        .def("growNonPeriodicDomain", nb::overload_cast<int>(&Geometry::growNonPeriodicDomain, nb::const_),
             nb::arg("ngrow"))
        .def("growPeriodicDomain", nb::overload_cast<IntVect const&>(&Geometry::growPeriodicDomain, nb::const_),
             nb::arg("ngrow"))
        .def("growPeriodicDomain", nb::overload_cast<int>(&Geometry::growPeriodicDomain, nb::const_),
             nb::arg("ngrow"))

        .def("setPeriodicity",
            &Geometry::setPeriodicity,
            nb::arg("period"),
            "Set periodicity flags and return the old flags.\n"
            "Note that, unlike Periodicity class, the flags are just boolean."
        )
        .def("coarsen", &Geometry::coarsen, nb::arg("rr"))
        .def("refine", &Geometry::refine, nb::arg("rr"))
        .def("outsideRoundOffDomain", nb::overload_cast<AMREX_D_DECL(ParticleReal, ParticleReal, ParticleReal)>
            (&Geometry::outsideRoundoffDomain, nb::const_),
            AMREX_D_DECL(nb::arg("x"), nb::arg("y"), nb::arg("z")),
            "Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")
        .def("insideRoundOffDomain", nb::overload_cast<AMREX_D_DECL(ParticleReal, ParticleReal, ParticleReal)>
            (&Geometry::insideRoundoffDomain, nb::const_),
            AMREX_D_DECL(nb::arg("x"), nb::arg("y"), nb::arg("z")),
            "Returns true if a point is inside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")

        // .def("computeRoundoffDomain")
    ;


    make_Vector<Geometry> (m, "Geometry");
}
