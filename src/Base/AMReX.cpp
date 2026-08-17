#include "pyAMReX.H"
#include "dlpack/DLPackHelpers.H"

#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{
    /** Guards amrex.initialize()/finalize() against concurrent entry.
     *
     * Both mutate process-global state (the AMReX instance stack, ParmParse,
     * the arenas, signal handlers), and the threading contract says to call
     * them from one thread.
     *
     * We refuse rather than wait. finalize() runs the garbage collector, which
     * on a free-threaded interpreter stops the world and waits for every
     * attached thread to reach a safe point -- a thread blocked in
     * lock() inside a pybind11 call is attached and never gets there, so
     * waiting would deadlock. Failing loudly also matches what the docs
     * promise: these are main-thread only.
     */
    std::mutex init_finalize_mutex;

    /** Thread currently inside initialize()/finalize(), if any.
     *
     * Tracked explicitly because try_lock() on a std::mutex the calling thread
     * already owns is undefined behaviour, and re-entry is reachable: finalize()
     * holds the lock across the garbage collector, which runs arbitrary Python
     * finalizers, and a __del__ that calls amrex.finalize() defensively lands
     * right back here on the same thread.
     */
    std::atomic<std::thread::id> init_finalize_owner{};

    /** Non-blocking lock; throws instead of waiting or re-entering. */
    class ScopedInitFinalizeLock
    {
    public:
        ScopedInitFinalizeLock ()
        {
            auto const self = std::this_thread::get_id();
            if (init_finalize_owner.load(std::memory_order_acquire) == self) {
                throw std::runtime_error(
                    "amrex.initialize()/finalize() cannot be called from "
                    "inside amrex.initialize()/finalize() -- most likely from "
                    "a __del__ run by the garbage collector during finalize()");
            }
            // try_lock() may fail spuriously per [thread.mutex.requirements.mutex],
            // so this can in principle refuse a genuinely single-threaded call.
            // Not retried on purpose: a retry loop is what the stop-the-world GC
            // deadlock above rules out, and a spurious failure raises a clear
            // error rather than corrupting the instance stack.
            if (!init_finalize_mutex.try_lock()) {
                throw std::runtime_error(
                    "amrex.initialize()/finalize() must be called from a "
                    "single thread; another thread is inside one of them");
            }
            init_finalize_owner.store(self, std::memory_order_release);
        }
        ~ScopedInitFinalizeLock ()
        {
            init_finalize_owner.store(std::thread::id{}, std::memory_order_release);
            init_finalize_mutex.unlock();
        }
        ScopedInitFinalizeLock (ScopedInitFinalizeLock const &) = delete;
        ScopedInitFinalizeLock & operator= (ScopedInitFinalizeLock const &) = delete;
        ScopedInitFinalizeLock (ScopedInitFinalizeLock &&) = delete;
        ScopedInitFinalizeLock & operator= (ScopedInitFinalizeLock &&) = delete;
    };
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
              ScopedInitFinalizeLock lock;

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
              ScopedInitFinalizeLock lock;
              prepare_finalize();
              amrex::Finalize();
          });
    m.def("finalize",
          [prepare_finalize](AMReX* pamrex) {
              ScopedInitFinalizeLock lock;
              prepare_finalize();
              amrex::Finalize(pamrex);
          });
}
