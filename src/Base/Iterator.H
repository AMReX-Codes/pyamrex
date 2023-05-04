/* Copyright 2021-2022 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <AMReX_Config.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_MultiFab.H>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>

namespace py = nanobind;
using namespace amrex;

namespace pyAMReX
{
    /** This is a helper function for the C++ equivalent of void operator++()
     *
     * In Python, iterators always are called with __next__, even for the
     * first access. This means we need to handle the first iterator element
     * explicitly, otherwise we will jump directly to the 2nd element. We do
     * this the same way as pybind11 does this, via a little state:
     *   https://github.com/AMReX-Codes/pyamrex/pull/50
     *   https://github.com/pybind/nanobind/blob/v2.10.0/include/nanobind/nanobind.h#L2269-L2282
     *
     * To avoid unnecessary (and expensive) copies, remember to only call this
     * helper always with py::return_value_policy::reference_internal!
     *
     *
     * @tparam T_Iterator This is usally MFIter or Par(Const)Iter or derived classes
     * @param it the current iterator
     * @return the updated iterator
     */
    template< typename T_Iterator >
    T_Iterator &
    iterator_next( T_Iterator & it )
    {
        py::object self = py::cast(it);
        if (!py::hasattr(self, "first_or_done"))
            self.attr("first_or_done") = true;

        bool first_or_done = self.attr("first_or_done").cast<bool>();
        if (first_or_done) {
            first_or_done = false;
            self.attr("first_or_done") = first_or_done;
        }
        else
            ++it;
        if( !it.isValid() )
        {
            first_or_done = true;
            it.Finalize();
            throw py::stop_iteration();
        }
        return it;
     }
}
