#include "pyAMReX.H"
#include "dlpack/DLPackHelpers.H"

#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

#include <mutex>
#include <stdexcept>
#include <string>

namespace
{
    /** Serializes amrex.initialize()/finalize().
     *
     * Both mutate process-global state (the AMReX instance stack, ParmParse,
     * the arenas, signal handlers). The threading contract says to call them
     * from one thread; this makes a violation a clean serialization rather
     * than a corrupted instance stack.
     */
    std::mutex init_finalize_mutex;
}

void init_AMReX(py::module& m)
{
    using namespace amrex;

    py::class_<AMReX>(m, "AMReX")
        .def_static("empty", &AMReX::empty)
        .def_static("size", &AMReX::size)
        .def_static("erase", &AMReX::erase)
        .def_static("top", &AMReX::top,
                    py::return_value_policy::reference)
    ;

    m.def("initialize",
          [](const py::list args) {
              std::scoped_lock lock(init_finalize_mutex);

              Vector<std::string> cargs{"amrex"};
              Vector<char*> argv;

              // Populate the "command line"
              for (const auto& v: args)
                  cargs.push_back(v.cast<std::string>());
              for (auto& v: cargs)
                  argv.push_back(&v[0]);
              int argc = argv.size();

              // note: +1 since there is an extra char-string array element,
              //       that ANSII C requires to be a simple NULL entry
              //       https://stackoverflow.com/a/39096006/2719194
              argv.push_back(NULL);
              char** tmp = argv.data();

              const bool build_parm_parse = (cargs.size() > 1);
              // TODO: handle version with MPI
              return Initialize(argc, tmp, build_parm_parse);
          }, py::return_value_policy::reference,
          "Initialize AMReX library");

    m.def("initialized", &Initialized,
          "Returns true if there are any currently-active and initialized "
          "AMReX instances (i.e. one for which amrex::Initialize has been called, "
          "and amrex::Finalize has not). Otherwise false.");
    m.def("size", &AMReX::size,
          "The amr stack size, the number of amr instances pushed.");

    constexpr auto run_gc = []() {
        // explicitly run the garbage collector, so deleted objects
        // get freed.
        // This is a convenience helper/bandage for making work with Python
        // garbage collectors in various implementations more easy.
        // https://github.com/AMReX-Codes/pyamrex/issues/81
        auto m_gc = py::module::import("gc");
        auto collect = m_gc.attr("collect");
        collect();
    };

    constexpr auto prepare_finalize = [run_gc]() {
        run_gc();
        auto const exports = pyAMReX::dlpack::outstanding_exports();
        if (exports != 0) {
            throw std::runtime_error(
                "amrex.finalize(): cannot finalize while " +
                std::to_string(exports) +
                " DLPack export(s) are still alive; delete all consuming "
                "arrays and capsules, then retry");
        }
    };

    m.def("finalize",
          [prepare_finalize]() {
              std::scoped_lock lock(init_finalize_mutex);
              prepare_finalize();
              amrex::Finalize();
          });
    m.def("finalize",
          [prepare_finalize](AMReX* pamrex) {
              std::scoped_lock lock(init_finalize_mutex);
              prepare_finalize();
              amrex::Finalize(pamrex);
          });
}
