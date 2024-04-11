/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FabFactory.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <string>

namespace {
    void check_comp(amrex::MultiFab const & mf, const int comp, std::string const name)
    {
        if (comp < 0 || comp >= mf.nComp())
            throw py::index_error("MultiFab::" + name + " comp out of bounds");
    }
    void check_nghost(amrex::MultiFab const & mf, const int nghost, std::string const name)
    {
        if (nghost < 0 || nghost > mf.nGrowVect().min())
            throw py::index_error("MultiFab::" + name + " nghost out of bounds");
    }
}

void init_MultiFab(py::module &m)
{
    using namespace amrex;

    py::class_< FabArrayBase > py_FabArrayBase(m, "FabArrayBase");
    py::class_< FabArray<FArrayBox>, FabArrayBase > py_FabArray_FArrayBox(m, "FabArray_FArrayBox");
    py::class_< MultiFab, FabArray<FArrayBox> > py_MultiFab(m, "MultiFab");

    py::class_< FabFactory<FArrayBox> >(m, "FabFactory_FArrayBox");

    py::class_< MFInfo >(m, "MFInfo")
        .def_readwrite("alloc", &MFInfo::alloc)
        .def_readwrite("arena", &MFInfo::arena)
        .def_readwrite("tags", &MFInfo::tags)

        .def(py::init< >())

        .def("set_alloc", &MFInfo::SetAlloc)
        .def("set_arena", &MFInfo::SetArena)
        //.def("set_tag", py::overload_cast< std::string >(&MFInfo::SetTag))
        .def("set_tag", [](MFInfo & info, std::string tag) { info.SetTag(std::move(tag)); })
    ;

    py::class_< MFItInfo >(m, "MFItInfo")
        .def_readwrite("do_tiling", &MFItInfo::do_tiling)
        .def_readwrite("dynamic", &MFItInfo::dynamic)
        .def_readwrite("device_sync", &MFItInfo::device_sync)
        .def_readwrite("num_streams", &MFItInfo::num_streams)
        .def_readwrite("tilesize", &MFItInfo::tilesize)

        .def(py::init< >())

        .def("enable_tiling", &MFItInfo::EnableTiling,
             py::arg("ts") /*=FabArrayBase::mfiter_tile_size*/ )
        .def("set_dynamic", &MFItInfo::SetDynamic,
             py::arg("f"))
        .def("disable_device_sync", &MFItInfo::DisableDeviceSync)
        .def("set_device_sync", &MFItInfo::SetDeviceSync,
             py::arg("f"))
        .def("set_num_streams", &MFItInfo::SetNumStreams,
             py::arg("n"))
        .def("use_default_stream", &MFItInfo::UseDefaultStream)
    ;

    py::class_< MFIter >(m, "MFIter", py::dynamic_attr())
        .def("__repr__",
             [](MFIter const & mfi) {
                 std::string r = "<amrex.MFIter (";
                 if( !mfi.isValid() ) { r.append("in"); }
                 r.append("valid)>");
                 return r;
             }
        )
        .def(py::init< FabArrayBase const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the FabArrayBase (argument 2) alive
             py::keep_alive<1, 2>()
        )
        .def(py::init< FabArrayBase const &, MFItInfo const & >())

        .def(py::init< MultiFab const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the MultiFab (argument 2) alive
            py::keep_alive<1, 2>()
        )
        .def(py::init< MultiFab const &, MFItInfo const & >())

        //.def(py::init< iMultiFab const & >())
        //.def(py::init< iMultiFab const &, MFItInfo const & >())

        // helpers for iteration __next__
        .def("_incr", &MFIter::operator++)
        .def("finalize", &MFIter::Finalize)

        .def("tilebox", py::overload_cast< >(&MFIter::tilebox, py::const_))
        .def("tilebox", py::overload_cast< IntVect const & >(&MFIter::tilebox, py::const_))
        .def("tilebox", py::overload_cast< IntVect const &, IntVect const & >(&MFIter::tilebox, py::const_))

        .def("validbox", &MFIter::validbox)
        .def("fabbox", &MFIter::fabbox)

        .def("nodaltilebox",
            py::overload_cast< int >(&MFIter::nodaltilebox, py::const_),
            py::arg("dir") = -1)

        .def("growntilebox",
            py::overload_cast< const IntVect& >(&MFIter::growntilebox, py::const_),
            py::arg("ng") = -1000000)

        .def("grownnodaltilebox",
            py::overload_cast< int, int >(&MFIter::grownnodaltilebox, py::const_),
            py::arg("int") = -1, py::arg("ng") = -1000000)
        .def("grownnodaltilebox",
            py::overload_cast< int, const IntVect& >(&MFIter::grownnodaltilebox, py::const_),
            py::arg("int"), py::arg("ng"))

        .def_property_readonly("is_valid", &MFIter::isValid)
        .def_property_readonly("index", &MFIter::index)
        .def_property_readonly("length", &MFIter::length)
    ;

    py_FabArrayBase
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

        .def_property_readonly("n_grow_vect", &FabArrayBase::nGrowVect,
            "Return the grow factor (per direction) that defines the region of definition.")
    ;

    py_FabArray_FArrayBox
        // define
        .def("clear", &FabArray<FArrayBox>::clear)
        .def("ok", &FabArray<FArrayBox>::ok)

        .def_property_readonly("arena", &FabArray<FArrayBox>::arena,
            "Provides access to the Arena this FabArray was build with.")
        .def_property_readonly("has_EB_fab_factory", &FabArray<FArrayBox>::hasEBFabFactory)
        .def_property_readonly("factory", &FabArray<FArrayBox>::Factory)

        //.def("array", py::overload_cast< const MFIter& >(&FabArray<FArrayBox>::array))
        //.def("const_array", &FabArray<FArrayBox>::const_array)
        .def("array", [](FabArray<FArrayBox> & fa, MFIter const & mfi)
            { return fa.array(mfi); },
            // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
            py::keep_alive<0, 1>()
        )
        .def("const_array", [](FabArray<FArrayBox> & fa, MFIter const & mfi)
            { return fa.const_array(mfi); },
            // as long as the return value (argument 0) exists, keep the fa (argument 1) alive
            py::keep_alive<0, 1>()
        )

        /* setters */
        .def("set_val",
            py::overload_cast< amrex::Real >(&FabArray<FArrayBox>::setVal<FArrayBox>),
            py::arg("val"),
            "Set all components in the entire region of each FAB to val."
        )
        .def("set_val",
             py::overload_cast< amrex::Real, int, int, int >(&FabArray<FArrayBox>::setVal<FArrayBox>),
             py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
             "Set the value of num_comp components in the valid region of\n"
             "each FAB in the FabArray, starting at component comp to val.\n"
             "Also set the value of nghost boundary cells."
        )
        .def("set_val",
             py::overload_cast< amrex::Real, int, int, IntVect const & >(&FabArray<FArrayBox>::setVal<FArrayBox>),
             py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost"),
             "Set the value of num_comp components in the valid region of\n"
             "each FAB in the FabArray, starting at component comp to val.\n"
             "Also set the value of nghost boundary cells."
        )
        .def("set_val",
             py::overload_cast< amrex::Real, Box const &, int, int, int >(&FabArray<FArrayBox>::setVal<FArrayBox>),
             py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
             "Set the value of num_comp components in the valid region of\n"
             "each FAB in the FabArray, starting at component comp, as well\n"
             "as nghost boundary cells, to val, provided they also intersect\n"
             "with the Box region."
        )
        .def("set_val",
             py::overload_cast< amrex::Real, Box const &, int, int, IntVect const & >(&FabArray<FArrayBox>::setVal<FArrayBox>),
             py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost"),
             "Set the value of num_comp components in the valid region of\n"
             "each FAB in the FabArray, starting at component comp, as well\n"
             "as nghost boundary cells, to val, provided they also intersect\n"
             "with the Box region."
        )

        .def("abs", py::overload_cast< int, int, int >(&FabArray<FArrayBox>::abs<FArrayBox>),
            py::arg("comp"), py::arg("ncomp"), py::arg("nghost")=0
        )
        .def("abs", py::overload_cast< int, int, IntVect const & >(&FabArray<FArrayBox>::abs<FArrayBox>),
             py::arg("comp"), py::arg("ncomp"), py::arg("nghost")
        )

        .def_static("saxpy",
            py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template Saxpy<FArrayBox>),
            py::arg("y"), py::arg("a"), py::arg("x"), py::arg("xcomp"), py::arg("ycomp"), py::arg("ncomp"), py::arg("nghost"),
            "y += a*x"
        )
        .def_static("xpay",
            py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template Xpay<FArrayBox>),
            py::arg("y"), py::arg("a"), py::arg("x"), py::arg("xcomp"), py::arg("ycomp"), py::arg("ncomp"), py::arg("nghost"),
            "y = x + a*y"
        )
        .def_static("lin_comb",
            py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template LinComb<FArrayBox>),
            py::arg("dst"),
            py::arg("a"), py::arg("x"), py::arg("xcomp"),
            py::arg("b"), py::arg("y"), py::arg("ycomp"),
            py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "dst = a*x + b*y"
        )

        .def("sum",
             py::overload_cast< int, IntVect const&, bool >(&FabArray<FArrayBox>::template sum<FArrayBox>, py::const_),
             py::arg("comp"), py::arg("nghost"), py::arg("local"),
             "Returns the sum of component \"comp\""
        )
        .def("sum_boundary",
            py::overload_cast< Periodicity const & >(&FabArray<FArrayBox>::SumBoundary),
            py::arg("period"),
            "Sum values in overlapped cells.  The destination is limited to valid cells."
        )
        .def("sum_boundary", py::overload_cast< int, int, Periodicity const & >(&FabArray<FArrayBox>::SumBoundary),
             py::arg("scomp"), py::arg("ncomp"), py::arg("period"),
             "Sum values in overlapped cells.  The destination is limited to valid cells."
        )
        .def("sum_boundary", py::overload_cast< int, int, IntVect const&, Periodicity const & >(&FabArray<FArrayBox>::SumBoundary),
             py::arg("scomp"), py::arg("ncomp"), py::arg("nghost"), py::arg("period"),
             "Sum values in overlapped cells.  The destination is limited to valid cells."
        )
        .def("sum_boundary", py::overload_cast< int, int, IntVect const&, IntVect const&, Periodicity const & >(&FabArray<FArrayBox>::SumBoundary),
             py::arg("scomp"), py::arg("ncomp"), py::arg("nghost"), py::arg("dst_nghost"), py::arg("period"),
             "Sum values in overlapped cells.  The destination is limited to valid cells."
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

    py_FabArray_FArrayBox
        .def("fill_boundary",
            py::overload_cast< bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("cross")=false,
            doc_fabarray_fillb
        )
        .def("fill_boundary",
            py::overload_cast< Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("period"),
            py::arg("cross")=false,
            doc_fabarray_fillb
        )
        .def("fill_boundary",
            py::overload_cast< IntVect const &, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("nghost"),
            py::arg("period"),
            py::arg("cross")=false,
            doc_fabarray_fillb
        )
        .def("fill_boundary",
            py::overload_cast< int, int, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("scomp"),
            py::arg("ncomp"),
            py::arg("cross")=false,
            doc_fabarray_fillb
        )
        .def("fill_boundary",
            py::overload_cast< int, int, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("scomp"),
            py::arg("ncomp"),
            py::arg("period"),
            py::arg("cross")=false,
            doc_fabarray_fillb
        )
        .def("fill_boundary",
            py::overload_cast< int, int, IntVect const &, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),
            py::arg("scomp"),
            py::arg("ncomp"),
            py::arg("nghost"),
            py::arg("period"),
            py::arg("cross")=false,
            doc_fabarray_fillb
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

    py_FabArray_FArrayBox
        .def("override_sync",
            py::overload_cast< Periodicity const & >(&FabArray<FArrayBox>::OverrideSync),
            py::arg("period"),
            doc_fabarray_osync
        )
        .def("override_sync",
             py::overload_cast< int, int, Periodicity const & >(&FabArray<FArrayBox>::OverrideSync),
             py::arg("scomp"), py::arg("ncomp"), py::arg("period"),
             doc_fabarray_osync
        )
    ;

    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"),
          "Copy from a host to device FabArray."
    );
    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
          "Copy from a host to device FabArray for a specific (number of) component(s)."
    );

    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"),
          "Copy from a device to host FabArray."
    );
    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp"),
          "Copy from a device to host FabArray for a specific (number of) component(s)."
    );

    py_MultiFab
        .def("__repr__",
             [](MultiFab const & mf) {
                 return "<amrex.MultiFab with '" + std::to_string(mf.nComp()) +
                        "' components>";
             }
        )

        /* Constructors */
        .def(py::init< >(),
            R"(Constructs an empty MultiFab.

            Data can be defined at a later time using the define member functions
            inherited from FabArray.)"
        )
        .def(py::init< Arena* >(),
            py::arg("a"),
            R"(Constructs an empty MultiFab.

            Data can be defined at a later time using the define member functions.
            If ``define`` is called later with a nullptr as MFInfo's arena, the
            default Arena ``a`` will be used.  If the arena in MFInfo is not a
            nullptr, the MFInfo's arena will be used.)"
        )
    ;

    constexpr auto doc_mf_init = R"(Constructs a MultiFab.

    The size of the FArrayBox is given by the Box grown by \p ngrow, and
    the number of components is given by \p ncomp. If \p info is set to
    not allocating memory, then no FArrayBoxes are allocated at
    this time but can be defined later.

    Parameters
    ----------
    bxs :
      a valid region
    dm :
      a DistribuionMapping
    ncomp :
      number of components
    ngrow :
      number of cells the region grows
    info :
      MultiFab info, including allocation Arena
    factory :
      FArrayBoxFactory for embedded boundaries)";

    py_MultiFab
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const &, FabFactory<FArrayBox> const & >(),
            py::arg("bxs"), py::arg("dm"), py::arg("ncomp"), py::arg("ngrow"),
            py::arg("info"), py::arg("factory"),
            doc_mf_init
        )
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int>(),
             py::arg("bxs"), py::arg("dm"), py::arg("ncomp"), py::arg("ngrow"),
             doc_mf_init
        )

        .def(py::init< const BoxArray&, const DistributionMapping&, int, IntVect const&,
                       MFInfo const&, FabFactory<FArrayBox> const & >(),
             py::arg("bxs"), py::arg("dm"), py::arg("ncomp"), py::arg("ngrow"),
             py::arg("info"), py::arg("factory"),
             doc_mf_init
        )
        .def(py::init< const BoxArray&, const DistributionMapping&, int, IntVect const&>(),
             py::arg("bxs"), py::arg("dm"), py::arg("ncomp"), py::arg("ngrow"),
             doc_mf_init
        )

        //.def(py::init< MultiFab const&, MakeType, int, int >())

        /* delayed defines */
        //.def("define",
        //    py::overload_cast< const BoxArray&, const DistributionMapping&, int, int,
        //                       MFInfo const &, FabFactory<FArrayBox> const &
        //>(&MultiFab::define))
        //.def("define",
        //    py::overload_cast< const BoxArray&, const DistributionMapping&, int,
        //                       IntVect const&, MFInfo const &, FabFactory<FArrayBox> const &
        //>(&MultiFab::define))

        /* sizes, etc. */
        .def("min",
             [](MultiFab const & mf, int comp, int nghost, bool local) {
                 check_comp(mf, comp, "min");
                 check_nghost(mf, nghost, "min");
                 return mf.min(comp, nghost, local); },
             py::arg("comp") = 0, py::arg("nghost") = 0, py::arg("local") = false,
             "Returns the minimum value of the specfied component of the MultiFab."
        )
        .def("min",
             [](MultiFab const & mf, Box const & region, int comp, int nghost, bool local) {
                 check_comp(mf, comp, "min");
                 check_nghost(mf, nghost, "min");
                 return mf.min(region, comp, nghost, local); },
             py::arg("region"), py::arg("comp") = 0, py::arg("nghost") = 0, py::arg("local") = false,
             "Returns the minimum value of the specfied component of the MultiFab over the region."
        )

        .def("max",
             [](MultiFab const & mf, int comp, int nghost, bool local) {
                 check_comp(mf, comp, "max");
                 check_nghost(mf, nghost, "max");
                 return mf.max(comp, nghost, local); },
             py::arg("comp") = 0, py::arg("nghost") = 0, py::arg("local") = false,
             "Returns the maximum value of the specfied component of the MultiFab."
        )
        .def("max",
             [](MultiFab const & mf, Box const & region, int comp, int nghost, bool local) {
                 check_comp(mf, comp, "max");
                 check_nghost(mf, nghost, "max");
                 return mf.max(region, comp, nghost, local); },
             py::arg("region"), py::arg("comp") = 0, py::arg("nghost") = 0, py::arg("local") = false,
             "Returns the maximum value of the specfied component of the MultiFab over the region."
             )

        .def("minIndex", &MultiFab::minIndex)
        .def("maxIndex", &MultiFab::maxIndex)

        /* norms */
        .def("norm0", py::overload_cast< int, int, bool, bool >(&MultiFab::norm0, py::const_))
        //.def("norm0", py::overload_cast< iMultiFab const &, int, int, bool >(&MultiFab::norm0, py::const_))

        .def("norminf",
             //py::overload_cast< int, int, bool, bool >(&MultiFab::norminf, py::const_)
             [](MultiFab const & mf, int comp, int nghost, bool local, bool ignore_covered) {
                 return mf.norminf(comp, nghost, local, ignore_covered);
             }
        )
        //.def("norminf", py::overload_cast< iMultiFab const &, int, int, bool >(&MultiFab::norminf, py::const_))

        .def("norm1", py::overload_cast< int, Periodicity const&, bool >(&MultiFab::norm1, py::const_))
        .def("norm1", py::overload_cast< int, int, bool >(&MultiFab::norm1, py::const_))
        .def("norm1", py::overload_cast< Vector<int> const &, int, bool >(&MultiFab::norm1, py::const_))

        .def("norm2", py::overload_cast< int >(&MultiFab::norm2, py::const_))
        .def("norm2", py::overload_cast< int, Periodicity const& >(&MultiFab::norm2, py::const_))
        .def("norm2", py::overload_cast< Vector<int> const & >(&MultiFab::norm2, py::const_))

        /* simple math */

        .def("sum",
             // py::overload_cast< int, bool >(&MultiFab::sum, py::const_),
             [](MultiFab const & mf, int comp , bool local) { return mf.sum(comp, local); },
             py::arg("comp") = 0, py::arg("local") = false,
             "Returns the sum of component 'comp' over the MultiFab -- no ghost cells are included."
        )
        .def("sum",
             // py::overload_cast< Box const &, int, bool >(&MultiFab::sum, py::const_),
             [](MultiFab const & mf, Box const & region, int comp , bool local) { return mf.sum(region, comp, local); },
             py::arg("region"), py::arg("comp") = 0, py::arg("local") = false,
             "Returns the sum of component 'comp' in the given 'region'. -- no ghost cells are included."
        )
        .def("sum_unique",
             py::overload_cast< int, bool, Periodicity const& >(&MultiFab::sum_unique, py::const_),
             py::arg("comp") = 0,
             py::arg("local") = false,
             py::arg_v("period", Periodicity::NonPeriodic(), "Periodicity.non_periodic()"),
             "Same as sum with local=false, but for non-cell-centered data, this"
             "skips non-unique points that are owned by multiple boxes."
        )
        .def("sum_unique",
             py::overload_cast< Box const&, int, bool >(&MultiFab::sum_unique, py::const_),
             py::arg("region"),
             py::arg("comp") = 0,
             py::arg("local") = false,
             "Returns the unique sum of component `comp` in the given "
             "region. Non-unique points owned by multiple boxes in the MultiFab are"
             "only added once. No ghost cells are included. This function does not take"
             "periodicity into account in the determination of uniqueness of points."
        )

        .def("plus",
            py::overload_cast< Real, int >(&MultiFab::plus),
            py::arg("val"), py::arg("nghost")=0,
            "Adds the scalar value val to the value of each cell in the\n"
            "valid region of each component of the MultiFab.  The value\n"
            "of nghost specifies the number of cells in the boundary\n"
            "region that should be modified."
        )
        .def("plus",
             py::overload_cast< Real, int, int, int >(&MultiFab::plus),
             py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
             "Adds the scalar value \\p val to the value of each cell in the\n"
             "specified subregion of the MultiFab.\n\n"
             "The subregion consists of the \\p num_comp components starting at component \\p comp.\n"
             "The value of nghost specifies the number of cells in the\n"
             "boundary region of each FArrayBox in the subregion that should\n"
             "be modified."
        )
        .def("plus",
             py::overload_cast< Real, const Box&, int >(&MultiFab::plus),
             py::arg("val"), py::arg("region"), py::arg("nghost")=0,
             "Adds the scalar value val to the value of each cell in the\n"
             "valid region of each component of the MultiFab, that also\n"
             "intersects the Box region.  The value of nghost specifies the\n"
             "number of cells in the boundary region of each FArrayBox in\n"
             "the subregion that should be modified."
        )
        .def("plus",
             py::overload_cast< Real, const Box&, int, int, int >(&MultiFab::plus),
             py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
             "Identical to the previous version of plus(), with the\n"
             "restriction that the subregion is further constrained to\n"
             "the intersection with Box region."
        )
        .def("plus",
            py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::plus),
            py::arg("mf"), py::arg("strt_comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "This function adds the values of the cells in mf to the corresponding\n"
            "cells of this MultiFab.  mf is required to have the same BoxArray or\n"
            "\"valid region\" as this MultiFab.  The addition is done only to num_comp\n"
            "components, starting with component number strt_comp.  The parameter\n"
            "nghost specifies the number of boundary cells that will be modified.\n"
            "If nghost == 0, only the valid region of each FArrayBox will be\n"
            "modified."
        )

        .def("minus",
            py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::minus),
            py::arg("mf"), py::arg("strt_comp"), py::arg("num_comp"), py::arg("nghost")=0,
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
            py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::divide),
            py::arg("mf"), py::arg("strt_comp"), py::arg("num_comp"), py::arg("nghost")=0,
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
            py::overload_cast< Real, int >(&MultiFab::mult),
            py::arg("val"), py::arg("nghost")=0,
            "Scales the value of each cell in the valid region of each\n"
            "component of the MultiFab by the scalar val (a[i] <- a[i]*val).\n"
            "The value of nghost specifies the number of cells in the\n"
            "boundary region that should be modified."
        )
        .def("mult",
            py::overload_cast< Real, int, int, int >(&MultiFab::mult),
            py::arg("val"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Scales the value of each cell in the specified subregion of the\n"
            "MultiFab by the scalar val (a[i] <- a[i]*val). The subregion\n"
            "consists of the num_comp components starting at component comp.\n"
            "The value of nghost specifies the number of cells in the\n"
            "boundary region of each FArrayBox in the subregion that should\n"
            "be modified."
        )
        .def("mult",
            py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::mult),
            py::arg("val"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Identical to the previous version of mult(), with the\n"
            "restriction that the subregion is further constrained to the\n"
            "intersection with Box region.  The value of nghost specifies the\n"
            "number of cells in the boundary region of each FArrayBox in\n"
            "the subregion that should be modified."
        )
        .def("mult",
             py::overload_cast< Real, Box const &, int >(&MultiFab::mult),
             py::arg("val"), py::arg("region"), py::arg("nghost")=0,
             "Scales the value of each cell in the valid region of each\n"
             "component of the MultiFab by the scalar val (a[i] <- a[i]*val),\n"
             "that also intersects the Box region.  The value of nghost\n"
             "specifies the number of cells in the boundary region of each\n"
             "FArrayBox in the subregion that should be modified."
        )

        .def("invert",
            py::overload_cast< Real, int >(&MultiFab::invert),
            py::arg("numerator"), py::arg("nghost"),
            "Replaces the value of each cell in the specified subregion of\n"
            "the MultiFab with its reciprocal multiplied by the value of\n"
            "numerator.  The value of nghost specifies the number of cells\n"
            "in the boundary region that should be modified."
        )
        .def("invert",
            py::overload_cast< Real, int, int, int >(&MultiFab::invert),
            py::arg("numerator"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Replaces the value of each cell in the specified subregion of\n"
            "the MultiFab with its reciprocal multiplied by the value of\n"
            "numerator. The subregion consists of the num_comp components\n"
            "starting at component comp.  The value of nghost specifies the\n"
            "number of cells in the boundary region of each FArrayBox in the\n"
            "subregion that should be modified."
        )
        .def("invert",
            py::overload_cast< Real, Box const &, int >(&MultiFab::invert),
            py::arg("numerator"), py::arg("region"), py::arg("nghost"),
            "Scales the value of each cell in the valid region of each\n"
            "component of the MultiFab by the scalar val (a[i] <- a[i]*val),\n"
            "that also intersects the Box region.  The value of nghost\n"
            "specifies the number of cells in the boundary region of each\n"
            "FArrayBox in the subregion that should be modified."
        )
        .def("invert",
            py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::invert),
            py::arg("numerator"), py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Identical to the previous version of invert(), with the\n"
            "restriction that the subregion is further constrained to the\n"
            "intersection with Box region.  The value of nghost specifies the\n"
            "number of cells in the boundary region of each FArrayBox in the\n"
            "subregion that should be modified."
        )

        .def("negate",
            py::overload_cast< int >(&MultiFab::negate),
            py::arg("nghost")=0,
            "Negates the value of each cell in the valid region of\n"
            "the MultiFab.  The value of nghost specifies the number of\n"
            "cells in the boundary region that should be modified."
        )
        .def("negate",
            py::overload_cast< int, int, int >(&MultiFab::negate),
            py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Negates the value of each cell in the specified subregion of\n"
            "the MultiFab.  The subregion consists of the num_comp\n"
            "components starting at component comp.  The value of nghost\n"
            "specifies the number of cells in the boundary region of each\n"
            "FArrayBox in the subregion that should be modified."
        )
        .def("negate",
            py::overload_cast< Box const &, int >(&MultiFab::negate),
            py::arg("region"), py::arg("nghost")=0,
            "Negates the value of each cell in the valid region of\n"
            "the MultiFab that also intersects the Box region.  The value\n"
            "of nghost specifies the number of cells in the boundary region\n"
            "that should be modified."
        )
        .def("negate",
            py::overload_cast< Box const &, int, int, int >(&MultiFab::negate),
            py::arg("region"), py::arg("comp"), py::arg("num_comp"), py::arg("nghost")=0,
            "Identical to the previous version of negate(), with the\n"
            "restriction that the subregion is further constrained to\n"
            "the intersection with Box region."
        )

        /* static (standalone) simple math functions */
        .def_static("dot",
            py::overload_cast< MultiFab const &, int, MultiFab const &, int, int, int, bool >(&MultiFab::Dot),
            py::arg("x"), py::arg("xcomp"),
            py::arg("y"), py::arg("ycomp"),
            py::arg("numcomp"), py::arg("nghost"), py::arg("local")=false,
            "Returns the dot product of two MultiFabs."
        )
        .def_static("dot",
            py::overload_cast< MultiFab const &, int, int, int, bool >(&MultiFab::Dot),
            py::arg("x"), py::arg("xcomp"),
            py::arg("numcomp"), py::arg("nghost"), py::arg("local")=false,
            "Returns the dot product of a MultiFab with itself."
        )
        //.def_static("dot", py::overload_cast< iMultiFab const&, const MultiFab&, int, MultiFab const&, int, int, int, bool >(&MultiFab::Dot))

        .def_static("add",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Add),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Add src to dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def_static("add",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Add),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Add src to dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )

        .def_static("subtract",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Subtract),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Subtract src from dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def_static("subtract",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Subtract),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Subtract src from dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )

        .def_static("multiply",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Multiply),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Multiply dst by src including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def_static("multiply",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Multiply),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Multiply dst by src including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )

        .def_static("divide",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Divide),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Divide dst by src including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )
        .def_static("divide",
            py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Divide),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Divide dst by src including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray."
        )

        .def_static("swap",
            py::overload_cast< MultiFab &, MultiFab &, int, int, int, int >(&MultiFab::Swap),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Swap from src to dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray.\n"
            "The swap is local."
        )
        .def_static("swap",
            py::overload_cast< MultiFab &, MultiFab &, int, int, int, IntVect const & >(&MultiFab::Swap),
            py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "Swap from src to dst including nghost ghost cells.\n"
            "The two MultiFabs MUST have the same underlying BoxArray.\n"
            "The swap is local."
        )

        .def_static("saxpy",
            // py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Saxpy)
            static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, int, int, int)>(&MultiFab::Saxpy),
            py::arg("dst"), py::arg("a"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "dst += a*src"
        )

        .def_static("xpay",
            // py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Xpay)
            static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, int, int, int)>(&MultiFab::Xpay),
            py::arg("dst"), py::arg("a"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "dst = src + a*dst"
        )

        .def_static("lin_comb",
            // py::overload_cast< MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, int >(&MultiFab::LinComb)
            static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, int)>(&MultiFab::LinComb),
            py::arg("dst"),
            py::arg("a"), py::arg("x"), py::arg("x_comp"),
            py::arg("b"), py::arg("y"), py::arg("y_comp"),
            py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "dst = a*x + b*y"
        )

        .def_static("add_product",
            py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, int >(&MultiFab::AddProduct),
            py::arg("dst"),
            py::arg("src1"), py::arg("comp1"),
            py::arg("src2"), py::arg("comp2"),
            py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"),
            "dst += src1*src2"
        )
        .def_static("add_product",
            py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::AddProduct),
            "dst += src1*src2"
        )

        /* simple data validity checks */
        .def("contains_nan",
            py::overload_cast< bool >(&MultiFab::contains_nan, py::const_),
            py::arg("local")=false
        )
        .def("contains_nan",
            py::overload_cast< int, int, int, bool >(&MultiFab::contains_nan, py::const_),
            py::arg("scomp"), py::arg("ncomp"), py::arg("ngrow")=0, py::arg("local")=false
        )
        .def("contains_nan",
            py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_nan, py::const_),
            py::arg("scomp"), py::arg("ncomp"), py::arg("ngrow"), py::arg("local")=false
        )

        .def("contains_inf",
            py::overload_cast< bool >(&MultiFab::contains_inf, py::const_),
            py::arg("local")=false
        )
        .def("contains_inf",
            py::overload_cast< int, int, int, bool >(&MultiFab::contains_inf, py::const_),
            py::arg("scomp"), py::arg("ncomp"), py::arg("ngrow")=0, py::arg("local")=false
        )
        .def("contains_inf",
            py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_inf, py::const_),
            py::arg("scomp"), py::arg("ncomp"), py::arg("ngrow"), py::arg("local")=false
        )

        .def("box_array", &MultiFab::boxArray)
        .def("dm", &MultiFab::DistributionMap)
        .def_property_readonly("n_comp", &MultiFab::nComp)
        .def_property_readonly("n_grow_vect", &MultiFab::nGrowVect)

        /* masks & ownership */
        // TODO:
        // - OverlapMask -> std::unique_ptr<MultiFab>
        // - OwnerMask -> std::unique_ptr<iMultiFab>

        /* Syncs */
        .def("average_sync", &MultiFab::AverageSync)
        .def("weighted_sync", &MultiFab::WeightedSync)
        //.def("override_sync", py::overload_cast< iMultiFab const &, Periodicity const & >(&MultiFab::OverrideSync))

        /* Init & Finalize */
        .def_static("initialize", &MultiFab::Initialize)
        .def_static("finalize", &MultiFab::Finalize)
    ;


    m.def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"))
     .def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"));
}
