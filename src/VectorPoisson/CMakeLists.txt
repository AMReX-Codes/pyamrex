# VectorPoisson module for pyAMReX

foreach(_dim 1 2 3)
    set(_target pyAMReX_${_dim}d)
    if(TARGET ${_target})
        target_sources(${_target}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/VectorPoissonSolver.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/VectorPoissonSolverNodal.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/PyVectorPoisson.cpp
        )
        target_include_directories(${_target}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endforeach()
