#include <pybind11/pybind11.h>
// #include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_CoordSys.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace amrex;

void init_Geometry(py::module& m)
{
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
        .def("__str",
             [](const Geometry& gm) {
                 std::stringstream s;
                 s << gm;
                 return s.str();
             })
        .def(py::init<>())
        // .def(py::init<const Box&, const RealBox*, 
                // int, int const*>())
        .def(py::init<
            const Box&,
            const RealBox&,
            int,
            Array<int, AMREX_SPACEDIM> const&
          >(),
          py::arg("dom"), py::arg("rb"), py::arg("coord"), py::arg("is_per"))

        // copy operators

        // data()
        // .def_property_readonly("dx",&Geometry::dx)

        // -------- using parmParse -------------
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
                                        (&Geometry::define), "Set geometry")
        

        .def("ProbDomain", py::overload_cast<>(&Geometry::ProbDomain, py::const_),
            "Return problem domain")
        // .def("ProbDomain", py::overload_cast<const RealBox&>(&Geometry::ProbDomain), 
        //     "Set problem domain")
        .def("ProbDomain", [](Geometry& gm, const RealBox& rb) { 
            if(gm.Ok()) { gm.ProbDomain(rb);}
            else { throw std::runtime_error("Can't call ProbDomain on undefined Geometry; use Define");}
        })
        .def("ProbLo", py::overload_cast<int>(&Geometry::ProbLo, py::const_), 
            "Get the lo end of the problem domain in specified direction")
        .def("ProbLo", 
            [](const Geometry& gm) { 
                Array<Real,AMREX_SPACEDIM> lo {{AMREX_D_DECL(gm.ProbLo(0),gm.ProbLo(1),gm.ProbLo(2))}};
                return lo;
            },
            "Get the list of lo ends of the problem domain"
        )
        .def("ProbHi", py::overload_cast<int>(&Geometry::ProbHi, py::const_), 
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
        .def("Domain", py::overload_cast<>(&Geometry::Domain, py::const_), "Return rectangular domain")
        // .def("Domain", py::overload_cast<const Box&>(&Geometry::Domain), "Set rectangular domain")
        .def("Domain", [](Geometry& gm, const Box& bx) {
            if(gm.Ok()) { gm.Domain(bx);}
            else { throw std::runtime_error("Can't call Domain on undefined Geometry; use Define");}})

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
        // .def("period", py::overload_cast<int>(&Geometry::period, py::const_),
        //     "Return the period in the specified direction")
        .def("period", [](const Geometry& gm, const int dir) { 
            if(gm.isPeriodic(dir)){ return gm.period(dir); }
            else { throw std::runtime_error("Geometry is not periodic in this direction."); }
        }, "Return the period in the specified direction")
        .def("periodicity", py::overload_cast<>(&Geometry::periodicity, py::const_)
            )
        .def("periodicity", py::overload_cast<const Box&>(&Geometry::periodicity, py::const_),
            "Return Periodicity object with lengths determined by input Box"
            )

        // .def("periodicShift", &Geometry::periodicShift)
        .def("growNonPeriodicDomain", &Geometry::growNonPeriodicDomain)
        .def("growPeriodicDomain", &Geometry::growPeriodicDomain)

        .def("setPeriodicity", &Geometry::setPeriodicity)
        .def("coarsen", &Geometry::coarsen)
        .def("refine", &Geometry::refine)
        .def("outsideRoundOffDomain", py::overload_cast<AMREX_D_DECL(Real, Real, Real)>
                (&Geometry::outsideRoundoffDomain, py::const_),
            "Returns true if a point is outside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")
        .def("insideRoundOffDomain", py::overload_cast<AMREX_D_DECL(Real, Real, Real)>
                (&Geometry::insideRoundoffDomain, py::const_),
            "Returns true if a point is inside the roundoff domain. All particles with positions inside the roundoff domain are sure to be mapped to cells inside the Domain() box. Note that the same need not be true for all points inside ProbDomain()")
        // .def("computeRoundoffDomain")
    ;

}
