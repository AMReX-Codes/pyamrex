/* Copyright 2024-2025 The AMReX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyAMReX.H"

#include <AMReX_AmrCore.H>
#include <AMReX_AmrMesh.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>
#include <AMReX_TagBox.H>
#include <AMReX_Vector.H>

#ifdef AMREX_PARTICLES
#   include <AMReX_AmrParGDB.H>
#endif

#include <memory>
#include <sstream>


namespace
{
    /** Trampoline class to allow Python subclasses of amrex::AmrCore.
     *
     * amrex::AmrCore declares five pure virtual member functions that a
     * concrete class must implement. This trampoline forwards each of them to
     * a Python override (if present). Codes such as ImpactX and WarpX subclass
     * AmrCore in C++; this enables the same pattern from Python.
     *
     * See pybind11 docs: "Overriding virtual functions in Python".
     *
     * Note: pyAMReX uses the classic std::unique_ptr holder (not smart_holder),
     * so this trampoline must NOT inherit py::trampoline_self_life_support. The
     * AmrCore instance is owned by the Python subclass object; lifetime of the
     * dependent AmrParGDB / ParticleContainer is handled via py::keep_alive.
     */
    struct PyAmrCore : public amrex::AmrCore
    {
        using amrex::AmrCore::AmrCore;  // inherit constructors

        // We use the _NAME variants so Python subclasses override using
        // snake_case method names, consistent with pyAMReX conventions.
        void ErrorEst (
            int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow
        ) override
        {
            PYBIND11_OVERRIDE_PURE_NAME(
                void, amrex::AmrCore, "error_est", ErrorEst,
                lev, tags, time, ngrow);
        }

        void MakeNewLevelFromScratch (
            int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
        ) override
        {
            PYBIND11_OVERRIDE_PURE_NAME(
                void, amrex::AmrCore, "make_new_level_from_scratch",
                MakeNewLevelFromScratch, lev, time, ba, dm);
        }

        void MakeNewLevelFromCoarse (
            int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
        ) override
        {
            PYBIND11_OVERRIDE_PURE_NAME(
                void, amrex::AmrCore, "make_new_level_from_coarse",
                MakeNewLevelFromCoarse, lev, time, ba, dm);
        }

        void RemakeLevel (
            int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm
        ) override
        {
            PYBIND11_OVERRIDE_PURE_NAME(
                void, amrex::AmrCore, "remake_level", RemakeLevel,
                lev, time, ba, dm);
        }

        void ClearLevel (int lev) override
        {
            PYBIND11_OVERRIDE_PURE_NAME(
                void, amrex::AmrCore, "clear_level", ClearLevel, lev);
        }
    };

    /** Class handle between declaration and method definition.
     *
     * The AmrCore type is declared first (init_AmrCore_class) so that types
     * referencing it in member function signatures (AmrParGDB, the particle
     * containers) render proper Python type names in their docstrings and
     * type stubs. The member functions are added later (init_AmrCore), once
     * the types AmrCore members reference (AmrParGDB) are registered, too.
     */
    std::unique_ptr< py::class_< amrex::AmrCore, amrex::AmrMesh, PyAmrCore > >
        py_AmrCore;
}


void init_AmrCore_class (py::module& m)
{
    using namespace amrex;

    py_AmrCore = std::make_unique<
        py::class_< AmrCore, AmrMesh, PyAmrCore > >(
            m, "AmrCore",
            R"pbdoc(
Base class for Python AMR applications that manage an AMReX mesh hierarchy.

Subclasses must implement ``make_new_level_from_scratch``,
``make_new_level_from_coarse``, ``remake_level``, ``clear_level`` and
``error_est``.  AMReX calls these Python overrides while creating or
regridding levels.
)pbdoc");
}

void init_AmrCore (py::module& /* m */)
{
    using namespace amrex;

    (*py_AmrCore)
        .def("__repr__",
            [](AmrCore const & amr_core) {
                std::stringstream s;
                s << amr_core.finestLevel();
                return "<amrex.AmrCore with finest level '" + s.str() + "'>";
            }
        )

        .def(py::init< >(),
             R"pbdoc(
Construct an empty AMR core.

The mesh metadata is read from AMReX runtime parameters when available.
)pbdoc")
        .def(py::init<
                const RealBox&,
                int,
                const Vector<int>&,
                int,
                Vector<IntVect> const&,
                Array<int, AMREX_SPACEDIM> const&
             >(),
             py::arg("rb"), py::arg("max_level_in"), py::arg("n_cell_in"),
             py::arg("coord"), py::arg("ref_ratios"), py::arg("is_per"),
             R"pbdoc(
Construct an AMR core from an explicit level-0 problem domain.

Parameters
----------
rb : RealBox
    Physical problem domain for level 0.
max_level_in : int
    Maximum AMR level to create.  Use 0 for a single-level hierarchy.
n_cell_in : Vector_int
    Number of level-0 cells in each coordinate direction.
coord : int
    AMReX coordinate-system identifier.
ref_ratios : Vector_IntVect
    Refinement ratio for each coarse level.  Its length is normally
    ``max_level_in``.
is_per : Sequence[int]
    Periodicity flags for each coordinate direction.
)pbdoc")
        .def(py::init< Geometry const&, AmrInfo const& >(),
             py::arg("level_0_geom"), py::arg("amr_info"),
             R"pbdoc(
Construct an AMR core from a level-0 geometry and an ``AmrInfo`` object.
)pbdoc")

        .def("init_from_scratch", &AmrCore::InitFromScratch, py::arg("time"),
             R"pbdoc(
Create the AMR hierarchy from scratch at simulation time ``time``.

This calls the Python overrides that allocate level data and, when
``max_level`` is greater than 0, calls ``error_est`` to create refined grids.
)pbdoc")
        .def("regrid", &AmrCore::regrid,
             py::arg("lbase"), py::arg("time"), py::arg("initial") = false,
             R"pbdoc(
Rebuild levels finer than ``lbase`` at simulation time ``time``.

``error_est`` is called to tag cells, followed by the level remake/create/clear
callbacks as needed.
)pbdoc")

#ifdef AMREX_PARTICLES
        // The AmrParGDB is owned by the AmrCore (unique_ptr m_gdb). We hand out
        // a non-owning reference; keep the AmrCore alive while it is used.
        .def("get_par_gdb", &AmrCore::GetParGDB,
             py::return_value_policy::reference_internal,
             R"pbdoc(
Return the particle geometry/database broker owned by this AMR core.

The returned ``AmrParGDB`` can be passed to particle-container constructors or
``define`` methods.  It is a non-owning view; the ``AmrCore`` is kept alive by
the binding while the broker is used from Python.
)pbdoc")
#endif
    ;

    // release the class handle held since init_AmrCore_class
    py_AmrCore.reset();
}
