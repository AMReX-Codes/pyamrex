# VectorPoisson module for pyAMReX

foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        VectorPoissonSolver.cpp
        VectorPoissonSolverNodal.cpp
        PyVectorPoisson.cpp
    )
endforeach()
