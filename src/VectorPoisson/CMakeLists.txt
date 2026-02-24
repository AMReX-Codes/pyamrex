# VectorPoisson module for pyAMReX

# Determine the target name based on dimensionality
if(AMReX_SPACEDIM EQUAL 1)
    set(PYAMREX_TARGET pyAMReX_1d)
elseif(AMReX_SPACEDIM EQUAL 2)
    set(PYAMREX_TARGET pyAMReX_2d)
elseif(AMReX_SPACEDIM EQUAL 3)
    set(PYAMREX_TARGET pyAMReX_3d)
endif()

# Add source files to the appropriate pyAMReX target
target_sources(${PYAMREX_TARGET}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/VectorPoissonSolver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VectorPoissonSolverNodal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PyVectorPoisson.cpp
)

target_include_directories(${PYAMREX_TARGET}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)