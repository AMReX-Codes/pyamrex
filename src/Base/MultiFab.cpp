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
        //.def(py::init< FabArrayBase const &, MFItInfo const & >())

        .def(py::init< MultiFab const & >(),
            // while the created iterator (argument 1: this) exists,
            // keep the MultiFab (argument 2) alive
            py::keep_alive<1, 2>()
        )
        //.def(py::init< MultiFab const &, MFItInfo const & >())

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

        .def_property_readonly("nComp", &FabArrayBase::nComp)
        .def_property_readonly("num_comp", &FabArrayBase::nComp)
        .def_property_readonly("size", &FabArrayBase::size)

        .def_property_readonly("n_grow_vect", &FabArrayBase::nGrowVect)
    ;

    py_FabArray_FArrayBox
        // define
        .def("clear", &FabArray<FArrayBox>::clear)
        .def("ok", &FabArray<FArrayBox>::ok)

        .def_property_readonly("arena", &FabArray<FArrayBox>::arena)
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

        .def_static("saxpy", py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template Saxpy<FArrayBox>)
        )
        .def_static("xpay", py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template Xpay<FArrayBox>))
        .def_static("lin_comb", py::overload_cast< FabArray<FArrayBox> &, Real, FabArray<FArrayBox> const &, int, Real, FabArray<FArrayBox> const &, int, int, int, IntVect const & >(&FabArray<FArrayBox>::template LinComb<FArrayBox>))

        .def("sum", py::overload_cast< int, IntVect const&, bool >(&FabArray<FArrayBox>::template sum<FArrayBox>, py::const_))
        .def("sum_boundary", py::overload_cast< Periodicity const & >(&FabArray<FArrayBox>::SumBoundary))
        .def("sum_boundary", py::overload_cast< int, int, Periodicity const & >(&FabArray<FArrayBox>::SumBoundary))
        .def("sum_boundary", py::overload_cast< int, int, IntVect const&, Periodicity const & >(&FabArray<FArrayBox>::SumBoundary))

        .def("fill_boundary", py::overload_cast< bool >(&FabArray<FArrayBox>::template FillBoundary<Real>), py::arg("cross")=false)
        .def("fill_boundary", py::overload_cast< Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>), py::arg("period"), py::arg("cross")=false)
        .def("fill_boundary", py::overload_cast< IntVect const &, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>), py::arg("nghost"), py::arg("period"), py::arg("cross")=false)
        .def("fill_boundary", py::overload_cast< int, int, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>), py::arg("scomp"), py::arg("ncomp"), py::arg("cross")=false)
        .def("fill_boundary", py::overload_cast< int, int, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>), py::arg("scomp"), py::arg("ncomp"), py::arg("period"), py::arg("cross")=false)
        .def("fill_boundary", py::overload_cast< int, int, IntVect const &, Periodicity const &, bool >(&FabArray<FArrayBox>::template FillBoundary<Real>),  py::arg("scomp"), py::arg("ncomp"), py::arg("nghost"), py::arg("period"), py::arg("cross")=false)

        /* Syncs */
        .def("override_sync", py::overload_cast< Periodicity const & >(&FabArray<FArrayBox>::OverrideSync))
    ;

    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src")
    );
    m.def("htod_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&htod_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp")
    );

    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const & >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src")
    );
    m.def("dtoh_memcpy",
          py::overload_cast< FabArray<FArrayBox> &, FabArray<FArrayBox> const &, int, int, int >(&dtoh_memcpy<FArrayBox>),
          py::arg("dest"), py::arg("src"), py::arg("scomp"), py::arg("dcomp"), py::arg("ncomp")
    );

    py_MultiFab
        .def("__repr__",
             [](MultiFab const & mf) {
                 return "<amrex.MultiFab with '" + std::to_string(mf.nComp()) +
                        "' components>";
             }
        )

        /* Constructors */
        .def(py::init< >())
        //.def(py::init< MultiFab && >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const & >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const &, FabFactory<FArrayBox> const & >())

        .def(py::init< const BoxArray&, const DistributionMapping&, int,
                       IntVect const& >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int,
                       IntVect const&,
                       MFInfo const& >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int,
                       IntVect const&,
                       MFInfo const&, FabFactory<FArrayBox> const & >())

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

        /* setters */
        //.def("set_val",
        //     py::overload_cast< Real >(&MultiFab::setVal))
        .def("set_val",
             [](MultiFab & mf, Real val) { mf.setVal(val); }
        )
        .def("set_val",
            [](MultiFab & mf, Real val, int comp, int num_comp) {
                mf.setVal(val, comp, num_comp);
            }
        )
        //.def("set_val",
        //     py::overload_cast< Real, int, int, int >(&MultiFab::setVal))
        .def("set_val",
             [](MultiFab & mf, Real val, int comp, int ncomp, int nghost) {
                mf.setVal(val, comp, ncomp, nghost);
            }
        )
        //.def("set_val",
        //     py::overload_cast< Real, int, int, IntVect const & >(&MultiFab::setVal))
        .def("set_val",
             [](MultiFab & mf, Real val, int comp, int ncomp, IntVect const & nghost) {
                 mf.setVal(val, comp, ncomp, nghost);
             }
        )

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
             //py::overload_cast< int, bool >(&MultiFab::sum, py::const_),
             [](MultiFab const & mf, int comp , bool local) { return mf.sum(comp, local); },
             py::arg("comp") = 0, py::arg("local") = false,
             "Returns the sum of component 'comp' over the MultiFab -- no ghost cells are included."
        )
        .def("sum_unique", &MultiFab::sum_unique,
             py::arg("comp") = 0,
             py::arg("local") = false,
             py::arg_v("period", Periodicity::NonPeriodic(), "Periodicity.non_periodic()"),
             "Same as sum with local=false, but for non-cell-centered data, this"
             "skips non-unique points that are owned by multiple boxes."
        )

        .def("abs",
            [](MultiFab & mf, int comp, int num_comp) { mf.abs(comp, num_comp); })
        //.def("abs", py::overload_cast< int, int, int >(&MultiFab::abs))
        .def("abs",
             [](MultiFab & mf, int comp, int num_comp, int nghost) { mf.abs(comp, num_comp, nghost); })

        .def("plus", py::overload_cast< Real, int >(&MultiFab::plus))
        .def("plus", [](MultiFab & mf, Real val, int comp, int num_comp) {
             mf.plus(val, comp, num_comp); })
        .def("plus", py::overload_cast< Real, int, int, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< Real, Box const &, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::plus))

        .def("minus", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::minus))

        // renamed: ImportError: overloading a method with both static and instance methods is not supported
        .def("divi", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::divide))

        .def("mult", py::overload_cast< Real, int >(&MultiFab::mult))
        .def("mult", [](MultiFab & mf, Real val, int comp, int num_comp) {
             mf.mult(val, comp, num_comp); })
        .def("mult", py::overload_cast< Real, int, int, int >(&MultiFab::mult))
        .def("mult", py::overload_cast< Real, Box const &, int >(&MultiFab::mult))
        .def("mult", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::mult))

        .def("invert", py::overload_cast< Real, int >(&MultiFab::invert))
        .def("invert", [](MultiFab & mf, Real val, int comp, int num_comp) {
             mf.invert(val, comp, num_comp); })
        .def("invert", py::overload_cast< Real, int, int, int >(&MultiFab::invert))
        .def("invert", py::overload_cast< Real, Box const &, int >(&MultiFab::invert))
        .def("invert", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::invert))

        .def("negate", py::overload_cast< int >(&MultiFab::negate))
        .def("negate", py::overload_cast< int, int, int >(&MultiFab::negate))
        .def("negate", py::overload_cast< Box const &, int >(&MultiFab::negate))
        .def("negate", py::overload_cast< Box const &, int, int, int >(&MultiFab::negate))

        /* static (standalone) simple math functions */
        .def_static("dot", py::overload_cast< MultiFab const &, int, MultiFab const &, int, int, int, bool >(&MultiFab::Dot))
        .def_static("dot", py::overload_cast< MultiFab const &, int, int, int, bool >(&MultiFab::Dot))
        //.def_static("dot", py::overload_cast< iMultiFab const&, const MultiFab&, int, MultiFab const&, int, int, int, bool >(&MultiFab::Dot))

        .def_static("add", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Add))
        .def_static("add", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Add))

        .def_static("subtract", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Subtract))
        .def_static("subtract", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Subtract))

        .def_static("multiply", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Multiply))
        .def_static("multiply", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Multiply))

        .def_static("divide", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Divide))
        .def_static("divide", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Divide))

        .def_static("swap", py::overload_cast< MultiFab &, MultiFab &, int, int, int, int >(&MultiFab::Swap))
        .def_static("swap", py::overload_cast< MultiFab &, MultiFab &, int, int, int, IntVect const & >(&MultiFab::Swap))

        .def_static("saxpy",
                    // py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Saxpy)
                    static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, int, int, int)>(&MultiFab::Saxpy)
        )

        .def_static("xpay",
                    // py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Xpay)
                    static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, int, int, int)>(&MultiFab::Xpay)
        )

        .def_static("lin_comb",
                    // py::overload_cast< MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, int >(&MultiFab::LinComb)
                    static_cast<void (*)(MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, int)>(&MultiFab::LinComb)
        )

        .def_static("add_product", py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, int >(&MultiFab::AddProduct))
        .def_static("add_product", py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::AddProduct))

        /* simple data validity checks */
        .def("contains_nan", py::overload_cast< bool >(&MultiFab::contains_nan, py::const_))
        .def("contains_nan", py::overload_cast< int, int, int, bool >(&MultiFab::contains_nan, py::const_))
        .def("contains_nan", py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_nan, py::const_))

        .def("contains_inf", py::overload_cast< bool >(&MultiFab::contains_inf, py::const_))
        .def("contains_inf", py::overload_cast< int, int, int, bool >(&MultiFab::contains_inf, py::const_))
        .def("contains_inf", py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_inf, py::const_))

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
        .def_static("initialize", &MultiFab::Initialize )
        .def_static("finalize", &MultiFab::Finalize )
    ;


    m.def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"))
     .def("copy_mfab", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Copy), py::arg("dst"), py::arg("src"), py::arg("srccomp"), py::arg("dstcomp"), py::arg("numcomp"), py::arg("nghost"));
}
