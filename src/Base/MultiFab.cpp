/* Copyright 2021 The AMReX Community
 *
 * Authors: Axel Huebl, ...
 * License: BSD-3-Clause-LBNL
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <string>

namespace py = pybind11;
using namespace amrex;

namespace {
    /** STL-like iteration over amrex::MFIter
     *
     * The amrex::MFIter interface is currently a bit too tricky to implement
     * std::begin() and std::end() safely with OpenMP threading.
     */
    class MFIterWrapper {
        std::shared_ptr<MFIter> mfi;
    public:
        explicit MFIterWrapper(const MultiFab& mf) {
            // For tiling support (OpenMP/thread pools) later on:
            // MFIter mfi(mf, TilingIfNotGPU());
            mfi = std::make_shared<MFIter>(mf);
        }
        std::shared_ptr<MFIter> operator*() { return mfi; }
        std::shared_ptr<MFIter const> operator*() const { return mfi; }
        MFIterWrapper& operator++() { ++(*mfi); return *this; }
    };

    class ValidSentinel {};

    bool operator==(MFIterWrapper const& it, ValidSentinel const&) {
        return (*it)->isValid();
    }
}

void init_MultiFab(py::module &m) {
    py::class_< MultiFab >(m, "MultiFab")
        .def("__repr__",
             [](MultiFab const & mf) {
                 return "<amrex.MultiFab with '" + std::to_string(mf.nComp()) +
                        "' components>";
             }
        )

        /* Constructors */
        .def(py::init< >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int, int,
                       MFInfo const &, FabFactory<FArrayBox>const & >())
        .def(py::init< const BoxArray&, const DistributionMapping&, int,
                       IntVect const&,
                       MFInfo const&, FabFactory<FArrayBox> const & >())
        .def(py::init< MultiFab const&, MakeType, int, int >())

        /* delayed defines */
        .def("define",
            py::overload_cast< const BoxArray&, const DistributionMapping&, int, int,
                               MFInfo const &, FabFactory<FArrayBox> const &
        >(&MultiFab::define))
        .def("define",
            py::overload_cast< const BoxArray&, const DistributionMapping&, int,
                               IntVect const&, MFInfo const &, FabFactory<FArrayBox> const &
        >(&MultiFab::define))

        /* sizes, etc. */
        .def("min",
             py::overload_cast< int, int, bool >(&MultiFab::min, py::const_))
        .def("min",
             py::overload_cast< Box const &, int, int, bool >(&MultiFab::min, py::const_))
        .def("max",
             py::overload_cast< int, int, bool >(&MultiFab::max, py::const_))
        .def("max",
             py::overload_cast< Box const &, int, int, bool >(&MultiFab::max, py::const_))
        .def("minIndex", &MultiFab::minIndex)
        .def("maxIndex", &MultiFab::maxIndex)

        /* norms */
        .def("norm0", py::overload_cast< int, int, bool, bool >(&MultiFab::norm0, py::const_))
        //.def("norm0", py::overload_cast< iMultiFab const &, int, int, bool >(&MultiFab::norm0, py::const_))

        .def("norminf", py::overload_cast< int, int, bool, bool >(&MultiFab::norminf, py::const_))
        //.def("norminf", py::overload_cast< iMultiFab const &, int, int, bool >(&MultiFab::norminf, py::const_))

        .def("norm1", py::overload_cast< int, Periodicity const&, bool >(&MultiFab::norm1, py::const_))
        .def("norm1", py::overload_cast< int, int, bool >(&MultiFab::norm1, py::const_))
        .def("norm1", py::overload_cast< Vector<int> const &, int, bool >(&MultiFab::norm1, py::const_))

        .def("norm2", py::overload_cast< int >(&MultiFab::norm2, py::const_))
        .def("norm2", py::overload_cast< int, Periodicity const& >(&MultiFab::norm2, py::const_))
        .def("norm2", py::overload_cast< Vector<int> const & >(&MultiFab::norm2, py::const_))

        /* simple math */
        .def("sum", &MultiFab::sum)

        .def("plus", py::overload_cast< Real, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< Real, int, int, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< Real, Box const &, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::plus))
        .def("plus", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::plus))

        .def("minus", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::minus))

        // renamed: ImportError: overloading a method with both static and instance methods is not supported
        .def("divi", py::overload_cast< MultiFab const &, int, int, int >(&MultiFab::divide))

        .def("mult", py::overload_cast< Real, int >(&MultiFab::mult))
        .def("mult", py::overload_cast< Real, int, int, int >(&MultiFab::mult))
        .def("mult", py::overload_cast< Real, Box const &, int >(&MultiFab::mult))
        .def("mult", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::mult))

        .def("invert", py::overload_cast< Real, int >(&MultiFab::invert))
        .def("invert", py::overload_cast< Real, int, int, int >(&MultiFab::invert))
        .def("invert", py::overload_cast< Real, Box const &, int >(&MultiFab::invert))
        .def("invert", py::overload_cast< Real, Box const &, int, int, int >(&MultiFab::invert))

        .def("negate", py::overload_cast< int >(&MultiFab::negate))
        .def("negate", py::overload_cast< int, int, int >(&MultiFab::negate))
        .def("negate", py::overload_cast< Box const &, int >(&MultiFab::negate))
        .def("negate", py::overload_cast< Box const &, int, int, int >(&MultiFab::negate))

        .def("sum_boundary", py::overload_cast< Periodicity const & >(&MultiFab::SumBoundary))
        .def("sum_boundary", py::overload_cast< int, int, Periodicity const & >(&MultiFab::SumBoundary))
        .def("sum_boundary", py::overload_cast< int, int, IntVect const&, Periodicity const & >(&MultiFab::SumBoundary))

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

        .def_static("copy", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, int >(&MultiFab::Copy))
        .def_static("copy", py::overload_cast< MultiFab &, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Copy))

        .def_static("swap", py::overload_cast< MultiFab &, MultiFab &, int, int, int, int >(&MultiFab::Swap))
        .def_static("swap", py::overload_cast< MultiFab &, MultiFab &, int, int, int, IntVect const & >(&MultiFab::Swap))

        .def_static("saxpy", py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Saxpy))
        .def_static("saxpy", py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Saxpy))

        .def_static("xpay", py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, int >(&MultiFab::Xpay))
        .def_static("xpay", py::overload_cast< MultiFab &, Real, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::Xpay))

        .def_static("lin_comb", py::overload_cast< MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, int >(&MultiFab::LinComb))
        .def_static("lin_comb", py::overload_cast< MultiFab &, Real, MultiFab const &, int, Real, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::LinComb))

        .def_static("add_product", py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, int >(&MultiFab::AddProduct))
        .def_static("add_product", py::overload_cast< MultiFab &, MultiFab const &, int, MultiFab const &, int, int, int, IntVect const & >(&MultiFab::AddProduct))

        /* simple data validity checks */
        .def("contains_nan", py::overload_cast< bool >(&MultiFab::contains_nan, py::const_))
        .def("contains_nan", py::overload_cast< int, int, int, bool >(&MultiFab::contains_nan, py::const_))
        .def("contains_nan", py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_nan, py::const_))

        .def("contains_inf", py::overload_cast< bool >(&MultiFab::contains_inf, py::const_))
        .def("contains_inf", py::overload_cast< int, int, int, bool >(&MultiFab::contains_inf, py::const_))
        .def("contains_inf", py::overload_cast< int, int, IntVect const &, bool >(&MultiFab::contains_inf, py::const_))

        /* masks & ownership */
        // TODO:
        // - OverlapMask -> std::unique_ptr<MultiFab>
        // - OwnerMask -> std::unique_ptr<iMultiFab>

        /* Syncs */
        .def("average_sync", &MultiFab::AverageSync)
        .def("weighted_sync", &MultiFab::WeightedSync)
        .def("override_sync", py::overload_cast< Periodicity const & >(&MultiFab::OverrideSync))
        //.def("override_sync", py::overload_cast< iMultiFab const &, Periodicity const & >(&MultiFab::OverrideSync))

        /* Init & Finalize */
        .def_static("initialize", &MultiFab::Initialize )
        .def_static("finalize", &MultiFab::Finalize )

        /* data access in Box index space */
        .def("__iter__",
            [](const MultiFab& mf) {
                return py::make_iterator(MFIterWrapper(mf), ValidSentinel{});
            },
            /* Essential: keep object alive while iterator exists */
            py::keep_alive<0, 1>()
            //py::return_value_policy::reference_internal
        )
    ;
}
