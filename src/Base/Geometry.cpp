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


void init_Geometry(py::module& m)
{
    using namespace amrex;

    py::class_<GeometryData>(m, "GeometryData")
        .def("__repr__",
            [](const GeometryData&) {
                return "<amrex.GeometryData>";
            }
        )
        .def(py::init<>())
        .def_readonly("prob_domain", &GeometryData::prob_domain, "The problem domain (real).")
        .def_readonly("domain", &GeometryData::domain, "The index domain.")
        .def_readonly("coord", &GeometryData::coord, "The Coordinates type.")
        .def_property_readonly("dx",
            [](const GeometryData& gd){
                std::array<Real,AMREX_SPACEDIM> dx {AMREX_D_DECL(
                    gd.dx[0], gd.dx[1], gd.dx[2]
                )};
                return dx;
            },
            "The cellsize for each coordinate direction."
        )
        .def_property_readonly("is_periodic",
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

    py::class_<Geometry, CoordSys>(m, "Geometry")
        .def("__repr__",
             [](py::object& obj) {
                 py::str py_name = obj.attr("__class__").attr("__name__");
                 const std::string name = py_name;
                 const auto gm = obj.cast<Geometry>();
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
        .def(py::init<>())
        .def(py::init<
            const Box&,
            const RealBox&,
            int,
            Array<int, AMREX_SPACEDIM> const&
          >(),
          py::arg("dom"), py::arg("rb"), py::arg("coord"), py::arg("is_per"))

        .def("data", &Geometry::data, "Returns non-static copy of geometry's stored data")
        // .def("setup")

        .def("ResetDefaultProbDomain",
            py::overload_cast<const RealBox&>
            (&Geometry::ResetDefaultProbDomain),
            "Reset default problem domain of Geometry class with a `RealBox`")
        .def("ResetDefaultPeriodicity",
            py::overload_cast<const Array<int,AMREX_SPACEDIM>& >
            (&Geometry::ResetDefaultPeriodicity),
            "Reset default periodicity of Geometry class with an Array of `int`")
        .def("ResetDefaultCoord",
            py::overload_cast< int >
            (&Geometry::ResetDefaultCoord),
            "Reset default coord of Geometry class with an Array of `int`")

        .def("define", py::overload_cast<const Box&, const RealBox&,
                                        int, Array<int,AMREX_SPACEDIM> const&>
                                        (&Geometry::define),
            py::arg("dom"), py::arg("rb"), py::arg("coord"), py::arg("is_per"),
            "Set geometry"
        )

        .def_property("prob_domain",
            py::overload_cast<>(&Geometry::ProbDomain, py::const_),
            py::overload_cast<RealBox const &>(&Geometry::ProbDomain),
            "The problem domain (real)."
        )
        .def("ProbLo", py::overload_cast<int>(&Geometry::ProbLo, py::const_),
            py::arg("dir"),
            "Get the lo end of the problem domain in specified direction")
        .def("ProbLo",
            [](const Geometry& gm) {
                Array<Real,AMREX_SPACEDIM> lo {{AMREX_D_DECL(gm.ProbLo(0),gm.ProbLo(1),gm.ProbLo(2))}};
                return lo;
            },
            "Get the list of lo ends of the problem domain"
        )
        .def("ProbHi", py::overload_cast<int>(&Geometry::ProbHi, py::const_),
             py::arg("dir"),
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

        .def_property("domain",
              py::overload_cast<>(&Geometry::Domain, py::const_),
              py::overload_cast<Box const &>(&Geometry::Domain),
              "The rectangular domain (index space)."
        )

        // GetVolume
        // .def("GetVolume", py::overload_cast<MultiFab&>(&Geometry::GetVolume, py::const_))
        // .def("GetVolume", py::overload_cast<)
        // ---- needs FArrayBox, BoxArray ! --------
        // GetDLogA
        // GetFaceArea

        .def("isPeriodic", py::overload_cast<int>(&Geometry::isPeriodic, py::const_),
            "Is the domain periodic in the specified direction?")
        .def("isAnyPeriodic", py::overload_cast<>(&Geometry::isAnyPeriodic, py::const_),
            "Is domain periodic in any direction?")
        .def("isAllPeriodic", py::overload_cast<>(&Geometry::isAllPeriodic, py::const_),
            "Is domain periodic in all directions?")
        .def("isPeriodic", py::overload_cast<>(&Geometry::isPeriodic, py::const_),
            "Return list indicating whether domain is periodic in each direction")
        .def("period",
            [](const Geometry& gm, const int dir) {
                if(gm.isPeriodic(dir)){ return gm.period(dir); }
                else { throw std::runtime_error("Geometry is not periodic in this direction."); }
            },
            py::arg("dir"),
            "Return the period in the specified direction")
        .def("periodicity",
            py::overload_cast<>(&Geometry::periodicity, py::const_)
        )
        .def("periodicity",
            py::overload_cast<const Box&>(&Geometry::periodicity, py::const_),
            py::arg("b"),
            "Return Periodicity object with lengths determined by input Box"
        )

        // .def("periodicShift", &Geometry::periodicShift)
        .def("growNonPeriodicDomain", py::overload_cast<IntVect const&>(&Geometry::growNonPeriodicDomain, py::const_),
            py::arg("ngrow"))
        .def("growNonPeriodicDomain", py::overload_cast<int>(&Geometry::growNonPeriodicDomain, py::const_),
             py::arg("ngrow"))
        .def("growPeriodicDomain", py::overload_cast<IntVect const&>(&Geometry::growPeriodicDomain, py::const_),
             py::arg("ngrow"))
        .def("growPeriodicDomain", py::overload_cast<int>(&Geometry::growPeriodicDomain, py::const_),
             py::arg("ngrow"))

        .def("setPeriodicity",
            &Geometry::setPeriodicity,
            py::arg("period"),
            "Set periodicity flags and return the old flags.\n"
            "Note that, unlike Periodicity class, the flags are just boolean."
        )
        .def("coarsen", &Geometry::coarsen, py::arg("rr"))
        .def("refine", &Geometry::refine, py::arg("rr"))
        .def("outsideRoundOffDomain", py::overload_cast<AMREX_D_DECL(ParticleReal, ParticleReal, ParticleReal)>
            (&Geometry::outsideRoundoffDomain, py::const_),
            AMREX_D_DECL(py::arg("x"), py::arg("y"), py::arg("z")),
            "Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")
        .def("insideRoundOffDomain", py::overload_cast<AMREX_D_DECL(ParticleReal, ParticleReal, ParticleReal)>
            (&Geometry::insideRoundoffDomain, py::const_),
            AMREX_D_DECL(py::arg("x"), py::arg("y"), py::arg("z")),
            "Returns true if a point is inside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")

        // .def("computeRoundoffDomain")
    ;


    make_Vector<Geometry> (m, "Geometry");
}
