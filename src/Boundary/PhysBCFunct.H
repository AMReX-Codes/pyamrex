/* Copyright 2026 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <functional>
#include <utility>


namespace pyAMReX
{
    /** A physical boundary condition functor calling back into Python.
     *
     * This fulfills the same interface as amrex::PhysBCFunct<F> at the
     * MultiFab level and can thus be passed to, e.g., the FillPatch
     * functions. The user-provided callable receives
     * (mf, dcomp, ncomp, nghost, time, bccomp) and is expected to fill
     * the ghost cells of mf that lie outside the physical domain,
     * e.g., for external Dirichlet (BCType.ext_dir) boundaries.
     *
     * Note: the callback runs on the host. When called from C++, the
     * pybind11 functional caster acquires the GIL automatically.
     */
    struct PhysBCFunctUser
    {
        using UserFillFunc =
            std::function<void(amrex::MultiFab&, int, int,
                               amrex::IntVect const&, amrex::Real, int)>;

        PhysBCFunctUser () = default;

        explicit PhysBCFunctUser (UserFillFunc f) : m_f(std::move(f)) {}

        void operator() (amrex::MultiFab& mf, int dcomp, int ncomp,
                         amrex::IntVect const& nghost, amrex::Real time,
                         int bccomp)
        {
            if (m_f) { m_f(mf, dcomp, ncomp, nghost, time, bccomp); }
        }

        UserFillFunc m_f;
    };
}
