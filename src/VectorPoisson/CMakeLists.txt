# VectorPoisson module for pyAMReX

foreach(D IN LISTS AMReX_SPACEDIM)
    set(_target pyAMReX_${D}d)
    if(TARGET ${_target})
        target_sources(${_target}
            PRIVATE
                VectorPoissonSolver.cpp
                VectorPoissonSolverNodal.cpp
                PyVectorPoisson.cpp
        )
        target_include_directories(${_target}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endforeach()