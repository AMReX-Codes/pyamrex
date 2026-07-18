/* Copyright 2021-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "pyAMReX.H"

#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArray.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <type_traits>


namespace
{
    template<typename T>
    void check_comp (T const & mf, std::string const type, int comp, std::string const name)
    {
        if (comp < 0 || comp >= mf.nComp()) {
            auto message = type + "::" + name + " comp out of bounds";
            throw nb::index_error(message.c_str());
        }
    }

    template<typename T>
    void check_nghost (T const & mf, std::string const type, int nghost, std::string const name)
    {
        if (nghost < 0 || nghost > mf.nGrowVect().min()) {
            auto message = type + "::" + name + " nghost out of bounds";
            throw nb::index_error(message.c_str());
        }
    }
}

/** Helper to create MultiFab and iMultiFab
 *
 * @tparam PY_T nb::class_ from nanobind
 * @param py_MultiFab the nanobind class
 * @param name either "MultiFab" or "iMultiFab"
 */
template<typename PY_T>
void make_MultiFab(PY_T & py_MultiFab, std::string const name)
{
    using namespace amrex;

    using T = typename PY_T::Type;              // MultiFab or iMultiFab
    using fab_type = typename T::fab_type;      // FArrayBox or IArrayBox
    using value_type = typename T::value_type;  // Real or int

    py_MultiFab
        .def("__repr__",
             [name](T const & mf) {
                 return "<amrex." + name + " with '" + std::to_string(mf.nComp()) +
                        "' components>";
             }
        )

        /* Constructors */
        .def(nb::init< >(),
            R"(Constructs an empty (i)MultiFab.

            Data can be defined at a later time using the define member functions
            inherited from FabArray.)"
        )
        .def(nb::init< Arena* >(),
             nb::arg("a"),
            R"(Constructs an empty (i)MultiFab.

            Data can be defined at a later time using the define member functions.
            If ``define`` is called later with a nullptr as MFInfo's arena, the
            default Arena ``a`` will be used.  If the arena in MFInfo is not a
            nullptr, the MFInfo's arena will be used.)"
        )
    ;

    constexpr auto const doc_mf_init = R"(Constructs an (i)MultiFab.

The size of the FArrayBox is given by the Box grown by \p ngrow, and
the number of components is given by \p ncomp. If \p info is set to
not allocating memory, then no FArrayBoxes are allocated at
this time but can be defined later.

Parameters
----------
bxs :
  a valid region
dm :
  a DistributionMapping
ncomp :
  number of components
ngrow :
  number of cells the region grows
info :
  (i)MultiFab info, including allocation Arena
factory :
  FArrayBoxFactory for embedded boundaries)";

    py_MultiFab
        .def(nb::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const &, FabFactory<fab_type> const & >(),
            nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
            nb::arg("info"), nb::arg("factory"),
            doc_mf_init
        )
        .def(nb::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const & >(),
             nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
             nb::arg("info"),
             doc_mf_init
        )
        .def(nb::init< const BoxArray&, const DistributionMapping&, int, int>(),
             nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
             doc_mf_init
        )

        .def(nb::init< const BoxArray&, const DistributionMapping&, int, IntVect const&,
                       MFInfo const& >(),
             nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
             nb::arg("info"),
             doc_mf_init
        )
        .def(nb::init< const BoxArray&, const DistributionMapping&, int, IntVect const&,
                       MFInfo const&, FabFactory<fab_type> const & >(),
            nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
            nb::arg("info"), nb::arg("factory"),
            doc_mf_init
        )
        .def(nb::init< const BoxArray&, const DistributionMapping&, int, IntVect const&>(),
             nb::arg("bxs"), nb::arg("dm"), nb::arg("ncomp"), nb::arg("ngrow"),
             doc_mf_init
        )

        //.def(nb::init< T const&, MakeType, int, int >())

        /* delayed defines */
        //.def("define",
        //    nb::overload_cast< const BoxArray&, const DistributionMapping&, int, int,
        //                       MFInfo const &, FabFactory<fab_type> const &
        //>(&T::define))
        //.def("define",
        //    nb::overload_cast< const BoxArray&, const DistributionMapping&, int,
        //                       IntVect const&, MFInfo const &, FabFactory<fab_type> const &
        //>(&T::define))

        /* sizes, etc. */
        .def("min",
             [name](T const & mf, int comp, int nghost, bool local)
             {
                 check_comp(mf, name, comp, "min");
                 check_nghost(mf, name, nghost, "min");
                 return mf.min(comp, nghost, local);
             },
             nb::arg("comp") = 0, nb::arg("nghost") = 0, nb::arg("local") = false,
             "Returns the minimum value of the specified component of the (i)MultiFab."
        )
        .def("min",
             [name](T const & mf, Box const & region, int comp, int nghost, bool local) {
                 check_comp(mf, name, comp, "min");
                 check_nghost(mf, name, nghost, "min");
                 return mf.min(region, comp, nghost, local); },
             nb::arg("region"), nb::arg("comp") = 0, nb::arg("nghost") = 0, nb::arg("local") = false,
             "Returns the minimum value of the specified component of the (i)MultiFab over the region."
        )

        .def("max",
             [name](T const & mf, int comp, int nghost, bool local) {
                 check_comp(mf, name, comp, "max");
                 check_nghost(mf, name, nghost, "max");
                 return mf.max(comp, nghost, local); },
             nb::arg("comp") = 0, nb::arg("nghost") = 0, nb::arg("local") = false,
             "Returns the maximum value of the specified component of the (i)MultiFab."
        )
        .def("max",
             [name](T const & mf, Box const & region, int comp, int nghost, bool local) {
                 check_comp(mf, name, comp, "max");
                 check_nghost(mf, name, nghost, "max");
                 return mf.max(region, comp, nghost, local); },
             nb::arg("region"), nb::arg("comp") = 0, nb::arg("nghost") = 0, nb::arg("local") = false,
             "Returns the maximum value of the specified component of the (i)MultiFab over the region."
        )

        .def("minIndex", &T::minIndex)
        .def("maxIndex", &T::maxIndex)
    ;

    /* norms */
    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
            .def("norm0", nb::overload_cast< int, int, bool, bool >(&T::norm0, nb::const_))
            .def("norm0", nb::overload_cast< iMultiFab const &, int, int, bool >(&T::norm0, nb::const_))

            .def("norminf",
                    //nb::overload_cast< int, int, bool, bool >(&T::norminf, nb::const_)
                 [](T const & mf, int comp, int nghost, bool local, bool ignore_covered) {
                     return mf.norminf(comp, nghost, local, ignore_covered);
                 }
            )
            //.def("norminf", nb::overload_cast< iMultiFab const &, int, int, bool >(&T::norminf, nb::const_))

            .def("norm1", nb::overload_cast< int, Periodicity const&, bool >(&T::norm1, nb::const_))
            .def("norm1", nb::overload_cast< int, int, bool >(&T::norm1, nb::const_))
            .def("norm1", nb::overload_cast< Vector<int> const &, int, bool >(&T::norm1, nb::const_))

            .def("norm2", nb::overload_cast< int >(&T::norm2, nb::const_))
            .def("norm2", nb::overload_cast< int, Periodicity const& >(&T::norm2, nb::const_))
            .def("norm2", nb::overload_cast< Vector<int> const & >(&T::norm2, nb::const_))
        ;
    }

    py_MultiFab
        /* simple math */
        .def("sum",
            // nb::overload_cast< int, bool >(&T::sum, nb::const_),
            [](T const & mf, int comp , bool local) { return mf.sum(comp, local); },
            nb::arg("comp") = 0, nb::arg("local") = false,
            "Returns the sum of component 'comp' over the (i)MultiFab -- no ghost cells are included."
        )
        .def("sum",
                // nb::overload_cast< Box const &, int, bool >(&T::sum, nb::const_),
             [](T const & mf, Box const & region, int comp , bool local) { return mf.sum(region, comp, local); },
             nb::arg("region"), nb::arg("comp") = 0, nb::arg("local") = false,
             "Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included."
        )
    ;

    // TODO: Missing in AMReX iMultiFab as of v25.02
    //       https://github.com/AMReX-Codes/amrex/issues/4317
    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
        .def("sum_unique",
             nb::overload_cast< int, bool, Periodicity const& >(&T::sum_unique, nb::const_),
             nb::arg("comp") = 0,
             nb::arg("local") = false,
             nb::arg("period") = Periodicity::NonPeriodic(),
             "Same as sum with local=false, but for non-cell-centered data, this"
             "skips non-unique points that are owned by multiple boxes."
        )
        .def("sum_unique",
             nb::overload_cast< Box const&, int, bool >(&T::sum_unique, nb::const_),
             nb::arg("region"),
             nb::arg("comp") = 0,
             nb::arg("local") = false,
             "Returns the unique sum of component `comp` in the given "
             "region. Non-unique points owned by multiple boxes in the MultiFab are"
             "only added once. No ghost cells are included. This function does not take"
             "periodicity into account in the determination of uniqueness of points."
        )
        ;
    }

    py_MultiFab
        .def("plus",
             nb::overload_cast< value_type, int >(&T::plus),
             nb::arg("val"), nb::arg("nghost")=0,
             "Adds the scalar value val to the value of each cell in the\n"
             "valid region of each component of the MultiFab.  The value\n"
             "of nghost specifies the number of cells in the boundary\n"
             "region that should be modified."
        )
        .def("plus",
             nb::overload_cast< value_type, int, int, int >(&T::plus),
             nb::arg("val"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Adds the scalar value \\p val to the value of each cell in the\n"
             "specified subregion of the MultiFab.\n\n"
             "The subregion consists of the \\p num_comp components starting at component \\p comp.\n"
             "The value of nghost specifies the number of cells in the\n"
             "boundary region of each FArrayBox in the subregion that should\n"
             "be modified."
        )
        .def("plus",
             nb::overload_cast< value_type, const Box&, int >(&T::plus),
             nb::arg("val"), nb::arg("region"), nb::arg("nghost")=0,
             "Adds the scalar value val to the value of each cell in the\n"
             "valid region of each component of the MultiFab, that also\n"
             "intersects the Box region.  The value of nghost specifies the\n"
             "number of cells in the boundary region of each FArrayBox in\n"
             "the subregion that should be modified."
        )
        .def("plus",
             nb::overload_cast< value_type, const Box&, int, int, int >(&T::plus),
             nb::arg("val"), nb::arg("region"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Identical to the previous version of plus(), with the\n"
             "restriction that the subregion is further constrained to\n"
             "the intersection with Box region."
        )
        .def("plus",
             nb::overload_cast< T const &, int, int, int >(&T::plus),
             nb::arg("mf"), nb::arg("strt_comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "This function adds the values of the cells in mf to the corresponding\n"
             "cells of this MultiFab.  mf is required to have the same BoxArray or\n"
             "\"valid region\" as this MultiFab.  The addition is done only to num_comp\n"
             "components, starting with component number strt_comp.  The parameter\n"
             "nghost specifies the number of boundary cells that will be modified.\n"
             "If nghost == 0, only the valid region of each FArrayBox will be\n"
             "modified."
        )

        .def("minus",
             nb::overload_cast< T const &, int, int, int >(&T::minus),
             nb::arg("mf"), nb::arg("strt_comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "This function subtracts the values of the cells in mf from the\n"
             "corresponding cells of this MultiFab.  mf is required to have the\n"
             "same BoxArray or \"valid region\" as this MultiFab.  The subtraction is\n"
             "done only to num_comp components, starting with component number\n"
             "strt_comp.  The parameter nghost specifies the number of boundary\n"
             "cells that will be modified.  If nghost == 0, only the valid region of\n"
             "each FArrayBox will be modified."
        )

                // renamed: ImportError: overloading a method with both static and instance methods is not supported
        .def("divi",
             nb::overload_cast< T const &, int, int, int >(&T::divide),
             nb::arg("mf"), nb::arg("strt_comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "This function divides the values of the cells in mf from the\n"
             "corresponding cells of this MultiFab.  mf is required to have the\n"
             "same BoxArray or \"valid region\" as this MultiFab.  The division is\n"
             "done only to num_comp components, starting with component number\n"
             "strt_comp.  The parameter nghost specifies the number of boundary\n"
             "cells that will be modified.  If nghost == 0, only the valid region of\n"
             "each FArrayBox will be modified.  Note, nothing is done to protect\n"
             "against divide by zero."
        )

        .def("mult",
             nb::overload_cast< value_type, int >(&T::mult),
             nb::arg("val"), nb::arg("nghost")=0,
             "Scales the value of each cell in the valid region of each\n"
             "component of the MultiFab by the scalar val (a[i] <- a[i]*val).\n"
             "The value of nghost specifies the number of cells in the\n"
             "boundary region that should be modified."
        )
        .def("mult",
             nb::overload_cast< value_type, int, int, int >(&T::mult),
             nb::arg("val"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Scales the value of each cell in the specified subregion of the\n"
             "MultiFab by the scalar val (a[i] <- a[i]*val). The subregion\n"
             "consists of the num_comp components starting at component comp.\n"
             "The value of nghost specifies the number of cells in the\n"
             "boundary region of each FArrayBox in the subregion that should\n"
             "be modified."
        )
        .def("mult",
             nb::overload_cast< value_type, Box const &, int, int, int >(&T::mult),
             nb::arg("val"), nb::arg("region"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Identical to the previous version of mult(), with the\n"
             "restriction that the subregion is further constrained to the\n"
             "intersection with Box region.  The value of nghost specifies the\n"
             "number of cells in the boundary region of each FArrayBox in\n"
             "the subregion that should be modified."
        )
        .def("mult",
             nb::overload_cast< value_type, Box const &, int >(&T::mult),
             nb::arg("val"), nb::arg("region"), nb::arg("nghost")=0,
             "Scales the value of each cell in the valid region of each\n"
             "component of the MultiFab by the scalar val (a[i] <- a[i]*val),\n"
             "that also intersects the Box region.  The value of nghost\n"
             "specifies the number of cells in the boundary region of each\n"
             "FArrayBox in the subregion that should be modified."
        )
    ;

    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
            .def("invert",
                 nb::overload_cast< value_type, int >(&T::invert),
                 nb::arg("numerator"), nb::arg("nghost"),
                 "Replaces the value of each cell in the specified subregion of\n"
                 "the MultiFab with its reciprocal multiplied by the value of\n"
                 "numerator.  The value of nghost specifies the number of cells\n"
                 "in the boundary region that should be modified."
            )
            .def("invert",
                 nb::overload_cast< value_type, int, int, int >(&T::invert),
                 nb::arg("numerator"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
                 "Replaces the value of each cell in the specified subregion of\n"
                 "the MultiFab with its reciprocal multiplied by the value of\n"
                 "numerator. The subregion consists of the num_comp components\n"
                 "starting at component comp.  The value of nghost specifies the\n"
                 "number of cells in the boundary region of each FArrayBox in the\n"
                 "subregion that should be modified."
            )
            .def("invert",
                 nb::overload_cast< value_type, Box const &, int >(&T::invert),
                 nb::arg("numerator"), nb::arg("region"), nb::arg("nghost"),
                 "Scales the value of each cell in the valid region of each\n"
                 "component of the MultiFab by the scalar val (a[i] <- a[i]*val),\n"
                 "that also intersects the Box region.  The value of nghost\n"
                 "specifies the number of cells in the boundary region of each\n"
                 "FArrayBox in the subregion that should be modified."
            )
            .def("invert",
                 nb::overload_cast< value_type, Box const &, int, int, int >(&T::invert),
                 nb::arg("numerator"), nb::arg("region"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
                 "Identical to the previous version of invert(), with the\n"
                 "restriction that the subregion is further constrained to the\n"
                 "intersection with Box region.  The value of nghost specifies the\n"
                 "number of cells in the boundary region of each FArrayBox in the\n"
                 "subregion that should be modified."
            )
        ;
    }

    py_MultiFab
        .def("negate",
             nb::overload_cast< int >(&T::negate),
             nb::arg("nghost")=0,
             "Negates the value of each cell in the valid region of\n"
             "the MultiFab.  The value of nghost specifies the number of\n"
             "cells in the boundary region that should be modified."
        )
        .def("negate",
             nb::overload_cast< int, int, int >(&T::negate),
             nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Negates the value of each cell in the specified subregion of\n"
             "the MultiFab.  The subregion consists of the num_comp\n"
             "components starting at component comp.  The value of nghost\n"
             "specifies the number of cells in the boundary region of each\n"
             "FArrayBox in the subregion that should be modified."
        )
        .def("negate",
             nb::overload_cast< Box const &, int >(&T::negate),
             nb::arg("region"), nb::arg("nghost")=0,
             "Negates the value of each cell in the valid region of\n"
             "the MultiFab that also intersects the Box region.  The value\n"
             "of nghost specifies the number of cells in the boundary region\n"
             "that should be modified."
        )
        .def("negate",
             nb::overload_cast< Box const &, int, int, int >(&T::negate),
             nb::arg("region"), nb::arg("comp"), nb::arg("num_comp"), nb::arg("nghost")=0,
             "Identical to the previous version of negate(), with the\n"
             "restriction that the subregion is further constrained to\n"
             "the intersection with Box region."
        )
    ;

    /* static (standalone) simple math functions */
    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
            .def("dot",
                 [](T const &self, int comp, T const &y, int y_comp, int numcomp, int nghost, bool local) {
                     return T::Dot(self, comp, y, y_comp, numcomp, nghost, local);
                 },
                 nb::arg("comp"),
                 nb::arg("y"), nb::arg("y_comp"),
                 nb::arg("numcomp"), nb::arg("nghost"), nb::arg("local") = false,
                 "Returns the dot product of self with another MultiFab."
            )
            .def("dot",
                 [](T const &self, int comp, int numcomp, int nghost, bool local) {
                     return T::Dot(self, comp, numcomp, nghost, local);
                 },
                 nb::arg("comp"),
                 nb::arg("numcomp"), nb::arg("nghost"), nb::arg("local") = false,
                 "Returns the dot product with itself."
            )
            .def("dot",
                 [](T const &self, const iMultiFab &mask, int comp, MultiFab const &y, int y_comp, int numcomp,
                    int nghost, bool local) {
                     return T::Dot(mask, self, comp, y, y_comp, numcomp, nghost, local);
                 },
                 nb::arg("mask"), nb::arg("comp"), nb::arg("y"), nb::arg("y_comp"),
                 nb::arg("numcomp"), nb::arg("nghost"), nb::arg("local") = false,
                 "Returns the dot product of self with another MultiFab where the mask is valid."
            )
        ;
    }

    py_MultiFab
        .def("add",
             [](T & self, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                 T::Add(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Add src to self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("add",
             [](T &self, T const &src, int srccomp, int comp, int numcomp, IntVect const &nghost) {
                 T::Add(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Add src to self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("copymf",
            [](T &self, T const &src, int srccomp, int dstcomp, int numcomp, int nghost) {
                T::Copy(self, src, srccomp, dstcomp, numcomp, nghost);
            },
            nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"),
            "Copy from src to self including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray. The copy is local"
        )
    ;

    // TODO: Missing in iMultiFab https://github.com/AMReX-Codes/amrex/issues/4317
    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
            .def("copymf",
            [](T &self, T const &src, int srccomp, int dstcomp, int numcomp, IntVect const &nghost) {
                T::Copy(self, src, srccomp, dstcomp, numcomp, nghost);
            },
            nb::arg("src"), nb::arg("srccomp"), nb::arg("dstcomp"), nb::arg("numcomp"), nb::arg("nghost"),
            "Copy from src to self including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray. The copy is local"
            )
        ;
    }

    py_MultiFab
        .def("subtract",
             [](T & self, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                 T::Subtract(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Subtract src from self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("subtract",
             [](T & self, T const & src, int srccomp, int comp, int numcomp, IntVect const & nghost) {
                 T::Subtract(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Subtract src from self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("multiply",
             [](T & self, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                 T::Multiply(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Multiply self by src including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("multiply",
             [](T &self, T const &src, int srccomp, int comp, int numcomp, IntVect const &nghost) {
                 T::Multiply(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Multiply self by src including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("divide",
             [](T & self, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                 T::Divide(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Divide self by src including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def("divide",  /* TODO: Missing in iMultiFab https://github.com/AMReX-Codes/amrex/issues/4317 */
             [](T &self, T const &src, int srccomp, int comp, int numcomp, IntVect const &nghost) {
                 T::Divide(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Divide self by src including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray."
        )

        .def("swap",  /* TODO: Missing in iMultiFab https://github.com/AMReX-Codes/amrex/issues/4317 */
             [](T &self, T &src, int srccomp, int comp, int numcomp, int nghost) {
                 T::Swap(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Swap from src to self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray.\n"
             "The swap is local."
        )
        .def("swap",  /* TODO: Missing in iMultiFab https://github.com/AMReX-Codes/amrex/issues/4317 */
             [](T &self, T &src, int srccomp, int comp, int numcomp, IntVect const &nghost) {
                 T::Swap(self, src, srccomp, comp, numcomp, nghost);
             },
             nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
             "Swap from src to self including nghost ghost cells.\n"
             "The two MultiFabs MUST have the same underlying BoxArray.\n"
             "The swap is local."
        )
    ;

    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
            .def("saxpy",
                 [](T & self, value_type a, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                     T::Saxpy(self, a, src, srccomp, comp, numcomp, nghost);
                 },
                 nb::arg("a"), nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
                 "self += a * src"
            )

            .def("xpay",
                 [](T & self, value_type a, T const & src, int srccomp, int comp, int numcomp, int nghost) {
                     T::Xpay(self, a, src, srccomp, comp, numcomp, nghost);
                 },
                 nb::arg("a"), nb::arg("src"), nb::arg("srccomp"), nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
                 "self = src + a * self"
            )

            .def("lin_comb",
                 [](T & self, value_type a, T const & x, int x_comp, value_type b, T const & y, int y_comp, int comp, int numcomp, int nghost) {
                     T::LinComb(self, a, x, x_comp, b, y, y_comp, comp, numcomp, nghost);
                 },
                 nb::arg("a"), nb::arg("x"), nb::arg("x_comp"),
                 nb::arg("b"), nb::arg("y"), nb::arg("y_comp"),
                 nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
                 "self = a * x + b * y"
            )

            .def("add_product",
                 [](T & self, MultiFab const & src1, int comp1, MultiFab const & src2, int comp2, int comp, int numcomp, int nghost) {
                     T::AddProduct(self, src1, comp1, src2, comp2, comp, numcomp, nghost);
                 },
                 nb::arg("src1"), nb::arg("comp1"),
                 nb::arg("src2"), nb::arg("comp2"),
                 nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
                 "self += src1 * src2"
            )
            .def("add_product",
                 [](T & self, MultiFab const & src1, int comp1, MultiFab const & src2, int comp2, int comp, int numcomp, IntVect const & nghost) {
                     T::AddProduct(self, src1, comp1, src2, comp2, comp, numcomp, nghost);
                 },
                 nb::arg("src1"), nb::arg("comp1"),
                 nb::arg("src2"), nb::arg("comp2"),
                 nb::arg("comp"), nb::arg("numcomp"), nb::arg("nghost"),
                 "self += src1 * src2"
            )

            /* simple data validity checks */
            .def("contains_nan",
                 nb::overload_cast< bool >(&T::contains_nan, nb::const_),
                 nb::arg("local")=false
            )
            .def("contains_nan",
                 nb::overload_cast< int, int, int, bool >(&T::contains_nan, nb::const_),
                 nb::arg("scomp"), nb::arg("ncomp"), nb::arg("ngrow")=0, nb::arg("local")=false
            )
            .def("contains_nan",
                 nb::overload_cast< int, int, IntVect const &, bool >(&T::contains_nan, nb::const_),
                 nb::arg("scomp"), nb::arg("ncomp"), nb::arg("ngrow"), nb::arg("local")=false
            )

            .def("contains_inf",
                 nb::overload_cast< bool >(&T::contains_inf, nb::const_),
                 nb::arg("local")=false
            )
            .def("contains_inf",
                 nb::overload_cast< int, int, int, bool >(&T::contains_inf, nb::const_),
                 nb::arg("scomp"), nb::arg("ncomp"), nb::arg("ngrow")=0, nb::arg("local")=false
            )
            .def("contains_inf",
                 nb::overload_cast< int, int, IntVect const &, bool >(&T::contains_inf, nb::const_),
                 nb::arg("scomp"), nb::arg("ncomp"), nb::arg("ngrow"), nb::arg("local")=false
            )
        ;
    }

    py_MultiFab
        .def("box_array", &T::boxArray)
        .def("dm", &T::DistributionMap)
        .def_prop_ro("n_comp", &T::nComp)
        .def_prop_ro("n_grow_vect", &T::nGrowVect)

        /* masks & ownership */
        // TODO:
        // - OverlapMask -> std::unique_ptr<MultiFab>
        // - OwnerMask -> std::unique_ptr<iMultiFab>
    ;

    /* Syncs */
    if constexpr (std::is_same_v<T, MultiFab>) {
        py_MultiFab
                .def("average_sync", &T::AverageSync)
                .def("weighted_sync", &T::WeightedSync)
                .def("override_sync", nb::overload_cast<iMultiFab const &, Periodicity const &>(&T::OverrideSync));
    }

    py_MultiFab
        /* Init & Finalize */
        .def_static("initialize", &T::Initialize)
        .def_static("finalize", &T::Finalize)
    ;
}
