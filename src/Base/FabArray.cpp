/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>

#include <memory>
#include <string>


namespace
{
    template<typename T>
    void make_FabArray_T(py::module &m, std::string const &name)
    {
        using namespace amrex;

        using FAT = FabArray<T>;
        using value_type = typename FAT::value_type;
        std::string const full_name = "FabArray_" + name;
        py::class_<FAT, FabArrayBase > py_FAT(m, full_name.c_str());
        py_FAT
            // define
            .def("clear", &FAT::clear)
            .def("ok", &FAT::ok)

            .def_property_readonly("arena", &FAT::arena,
                                   "Provides access to the Arena this FabArray was build with.")
            .def_property_readonly("has_EB_fab_factory", &FAT::hasEBFabFactory)
            .def_property_readonly("factory", &FAT::Factory)

            //.def("array", py::overload_cast< const MFIter& >(&FAT::array))
            //.def("const_array", &FAT::const_array)
            .def("array", [](FAT & fa, MFIter const & mfi)
                 { return fa.array(mfi); },
                    // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
                 py::keep_alive<0, 1>()
            )
            .def("const_array", [](FAT & fa, MFIter const & mfi)
                 { return fa.const_array(mfi); },
                    // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
                 py::keep_alive<0, 1>()
            )

            /* setters */
            .def("set_val",
                 py::overload_cast< value_type >(&FAT::template setVal<T>),
                 py::arg("val"),
                 "Set all components in the entire region of each FAB to val."
            )
            .def("set_val",
                 py::overload_cast< value_type, int, int, int >(&FAT::template setVal<T>),
                 py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
                 "Set the value of num_comp components in the valid region of\n"
                 "each FAB in the FabArray, starting at component comp to val.\n"
                 "Also set the value of nghost boundary cells."
            )
            .def("set_val",
                 py::overload_cast< value_type, int, int, IntVect const & >(&FAT::template setVal<T>),
                 py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost"),
                 "Set the value of num_comp components in the valid region of\n"
                 "each FAB in the FabArray, starting at component comp to val.\n"
                 "Also set the value of nghost boundary cells."
            )
            .def("set_val",
                 py::overload_cast< value_type, Box const &, int, int, int >(&FAT::template setVal<T>),
                 py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
                 "Set the value of num_comp components in the valid region of\n"
                 "each FAB in the FabArray, starting at component comp, as well\n"
                 "as nghost boundary cells, to val, provided they also intersect\n"
                 "with the Box region."
            )
            .def("set_val",
                 py::overload_cast< value_type, Box const &, int, int, IntVect const & >(&FAT::template setVal<T>),
                 py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost"),
                 "Set the value of num_comp components in the valid region of\n"
                 "each FAB in the FabArray, starting at component comp, as well\n"
                 "as nghost boundary cells, to val, provided they also intersect\n"
                 "with the Box region."
            )

            .def("abs", py::overload_cast< int, int, int >(&FAT::template abs<T>),
                 py::arg("comp"), py::arg("ncomp"), py::arg("nghost")=0
            )
            .def("abs", py::overload_cast< int, int, IntVect const & >(&FAT::template abs<T>),
                 py::arg("comp"), py::arg("ncomp"), py::arg("nghost")
            )

            .def("saxpy",
                 [](FAT & dst, value_type a, FAT const & x, int x_comp, int comp, int ncomp, IntVect const & nghost)
                 {
                     FAT::Saxpy(dst, a, x, x_comp, comp, ncomp, nghost);
                 },
                 py::arg("a"), py::arg("x"), py::arg("x_comp"), py::arg("comp"), py::arg("ncomp"), py::arg("nghost"),
                 "self += a * x\n\n"
                 "Parameters\n"
                 "----------\n"
                 "a      : scalar a\n"
                 "x      : FabArray x\n"
                 "x_comp : starting component of x\n"
                 "comp   : starting component of self\n"
                 "ncomp  : number of components\n"
                 "nghost : number of ghost cells"
            )
            .def("xpay",
                 [](FAT & self, value_type a, FAT const & x, int x_comp, int comp, int ncomp, IntVect const & nghost)
                 {
                     FAT::Xpay(self, a, x, x_comp, comp, ncomp, nghost);
                 },
                 py::arg("a"), py::arg("x"), py::arg("xcomp"), py::arg("comp"), py::arg("ncomp"), py::arg("nghost"),
                 "self = x + a * self\n\n"
                 "Parameters\n"
                 "----------\n"
                 "a      : scalar a\n"
                 "x      : FabArray x\n"
                 "x_comp : starting component of x\n"
                 "comp   : starting component of self\n"
                 "ncomp  : number of components\n"
                 "nghost : number of ghost cells"
            )
            .def("lin_comb",
                 [](
                         FAT & dst,
                         value_type a, FAT const & x, int x_comp,
                         value_type b, FAT const & y, int y_comp,
                         int comp, int ncomp, IntVect const & nghost)
                 {
                     FAT::LinComb(dst, a, x, x_comp, b, y, y_comp, comp, ncomp, nghost);
                 },
                 py::arg("a"), py::arg("x"), py::arg("xcomp"),
                 py::arg("b"), py::arg("y"), py::arg("ycomp"),
                 py::arg("comp"), py::arg("numcomp"), py::arg("nghost"),
                 "self = a * x + b * y\n\n"
                 "Parameters\n"
                 "----------\n"
                 "a     : float\n"
                 "    scalar a\n"
                 "x     : FabArray\n"
                 "xcomp : int\n"
                 "    starting component of x\n"
                 "b     : float\n"
                 "    scalar b\n"
                 "y     : FabArray\n"
                 "ycomp : int\n"
                 "    starting component of y\n"
                 "comp  : int\n"
                 "    starting component of self\n"
                 "numcomp : int\n"
                 "    number of components\n"
                 "nghost  : int\n"
                 "    number of ghost cells"
            )

            .def("sum",
                 py::overload_cast< int, IntVect const&, bool >(&FAT::template sum<T>, py::const_),
                 py::arg("comp"), py::arg("nghost"), py::arg("local"),
                 "Returns the sum of component \"comp\""
            )
            .def("sum_boundary",
                 py::overload_cast< Periodicity const &, bool >(&FAT::SumBoundary),
                 py::arg("period"), py::arg("deterministic") = false,
                 "Sum values in overlapped cells.  The destination is limited to valid cells."
            )
            .def("sum_boundary", py::overload_cast< int, int, Periodicity const &, bool >(&FAT::SumBoundary),
                 py::arg("scomp"), py::arg("ncomp"), py::arg("period"), py::arg("deterministic") = false,
                 "Sum values in overlapped cells.  The destination is limited to valid cells."
            )
            .def("sum_boundary", py::overload_cast< int, int, IntVect const&, Periodicity const &, bool >(&FAT::SumBoundary),
                 py::arg("scomp"), py::arg("ncomp"), py::arg("nghost"), py::arg("period"),
                 py::arg("deterministic") = false,
                 "Sum values in overlapped cells.  The destination is limited to valid cells."
            )
            .def("sum_boundary", py::overload_cast< int, int, IntVect const&, IntVect const&, Periodicity const &, bool >(&FAT::SumBoundary),
                 py::arg("scomp"), py::arg("ncomp"), py::arg("nghost"), py::arg("dst_nghost"), py::arg("period"),
                 py::arg("deterministic") = false,
                 "Sum values in overlapped cells.  The destination is limited to valid cells."
            )
        ;

        constexpr auto doc_fabarray_osync = R"(Synchronize nodal data.

    The synchronization will override valid regions by the intersecting valid regions with a higher precedence.
    The smaller the global box index is, the higher precedence the box has.
    With periodic boundaries, for cells in the same box, those near the lower corner have higher precedence than those near the upper corner.

    Parameters
    ----------
    scomp :
      starting component
    ncomp :
      number of components
    period :
      periodic length if it's non-zero)";

        py_FAT
            .def("override_sync",
                 py::overload_cast< Periodicity const & >(&FAT::OverrideSync),
                 py::arg("period"),
                 doc_fabarray_osync
            )
            .def("override_sync",
                 py::overload_cast< int, int, Periodicity const & >(&FAT::OverrideSync),
                 py::arg("scomp"), py::arg("ncomp"), py::arg("period"),
                 doc_fabarray_osync
            )
        ;

        constexpr auto doc_fabarray_fillb = R"(Copy on intersection within a FabArray.

    Data is copied from valid regions to intersecting regions of definition.
    The purpose is to fill in the boundary regions of each FAB in the FabArray.
    If cross=true, corner cells are not filled. If the length of periodic is provided,
    periodic boundaries are also filled.

    If scomp is provided, this only copies ncomp components starting at scomp.

    Note that FabArray itself does not contains any periodicity information.
    FillBoundary expects that its cell-centered version of its BoxArray is non-overlapping.)";

        py_FAT
            .def("fill_boundary",
                 py::overload_cast< bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
            .def("fill_boundary",
                 py::overload_cast< Periodicity const &, bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("period"),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
            .def("fill_boundary",
                 py::overload_cast< IntVect const &, Periodicity const &, bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("nghost"),
                 py::arg("period"),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
            .def("fill_boundary",
                 py::overload_cast< int, int, bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("scomp"),
                 py::arg("ncomp"),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
            .def("fill_boundary",
                 py::overload_cast< int, int, Periodicity const &, bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("scomp"),
                 py::arg("ncomp"),
                 py::arg("period"),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
            .def("fill_boundary",
                 py::overload_cast< int, int, IntVect const &, Periodicity const &, bool >(&FAT::template FillBoundary<value_type>),
                 py::arg("scomp"),
                 py::arg("ncomp"),
                 py::arg("nghost"),
                 py::arg("period"),
                 py::arg("cross")=false,
                 doc_fabarray_fillb
            )
        ;
    }
}

void
init_FabArray(py::module &m)
{
    using namespace amrex;

    py::class_< FabArrayBase >(m, "FabArrayBase")
        .def_property_readonly("is_all_cell_centered", &FabArrayBase::is_cell_centered)
        .def_property_readonly("is_all_nodal",
             py::overload_cast< >(&FabArrayBase::is_nodal, py::const_))
        .def("is_nodal",
             py::overload_cast< int >(&FabArrayBase::is_nodal, py::const_))

        .def_property_readonly("nComp", &FabArrayBase::nComp,
            "Return number of variables (aka components) associated with each point.")
        .def_property_readonly("num_comp", &FabArrayBase::nComp,
            "Return number of variables (aka components) associated with each point.")
        .def_property_readonly("size", &FabArrayBase::size,
            "Return the number of FABs in the FabArray.")
        .def("__len__", &FabArrayBase::size,
            "Return the number of FABs in the FabArray.")

        .def_property_readonly("n_grow_vect", &FabArrayBase::nGrowVect,
            "Return the grow factor (per direction) that defines the region of definition.")
    ;

    py::class_< FabFactory<IArrayBox> >(m, "FabFactory_IArrayBox");
    py::class_< FabFactory<FArrayBox> >(m, "FabFactory_FArrayBox");

    make_FabArray_T<IArrayBox>(m, "IArrayBox");
    make_FabArray_T<FArrayBox>(m, "FArrayBox");
}
