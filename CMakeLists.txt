# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.24.0)
project(pyAMReX VERSION 25.01)

include(${pyAMReX_SOURCE_DIR}/cmake/pyAMReXFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# Setting a cmake_policy to OLD is deprecated by definition and will raise a
# verbose warning
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
endif()

# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()
# device link step not yet fully implemented in AMReX logic
# https://cmake.org/cmake/help/latest/policy/CMP0105.html
if(POLICY CMP0105)
    cmake_policy(SET CMP0105 OLD)
endif()


# C++ Standard in Superbuilds #################################################
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
set_cxx17_superbuild()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(pyAMReX_CCACHE_DEFAULT ON)
else()
    set(pyAMReX_CCACHE_DEFAULT OFF)  # we are a subproject in a superbuild
endif()
option(pyAMReX_CCACHE "Enable ccache for faster rebuilds" ${pyAMReX_CCACHE_DEFAULT})
if(pyAMReX_CCACHE)
    set_ccache()
endif()


# Output (build) Directories ##################################################
#
# temporary build directories
pyamrex_set_default_build_dirs()

# default installation directories (w/o Python)
pyamrex_set_default_install_dirs()


# Options and Variants ########################################################
#
set(_pyAMReX_IPO_DEFAULT ON)
if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(_pyAMReX_IPO_DEFAULT ${CMAKE_INTERPROCEDURAL_OPTIMIZATION})
endif()
option(pyAMReX_IPO
    "Compile with interprocedural optimization (IPO) / link-time optimization (LTO)"
    ${_pyAMReX_IPO_DEFAULT}
)
option(pyAMReX_INSTALL
    "Enable install targets for pyAMReX"
    ON
)

# change the default build type to Release (or RelWithDebInfo) instead of Debug
set_default_build_type("Release")

# Testing logic with possibility to overwrite on a project basis in superbuilds
set_testing_option(pyAMReX_BUILD_TESTING)  # default: ON (BUILD_TESTING)


# Dependencies ################################################################
#

# AMReX
#   builds AMReX from source (default) or finds an existing install
include(${pyAMReX_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)
include(AMReXBuildInfo)  # <AMReX_buildInfo.H>

# for some targets need to be triggered once, so any dim dependency will do
list(LENGTH AMReX_SPACEDIM list_len)
math(EXPR list_last "${list_len} - 1")
list(GET AMReX_SPACEDIM ${list_last} AMReX_SPACEDIM_LAST)

# Python
find_package(Python 3.8.0 COMPONENTS Interpreter Development.Module REQUIRED)

# default installation directories: Python
pyamrex_set_default_install_dirs_python()

# pybind11
#   builds pybind11 from git (default), form local source or
#   finds an existing install
include(${pyAMReX_SOURCE_DIR}/cmake/dependencies/pybind11.cmake)


# Targets #####################################################################
#
foreach(D IN LISTS AMReX_SPACEDIM)
    # collect all objects for compilation
    add_library(pyAMReX_${D}d MODULE src/pyAMReX.cpp)
    add_library(pyAMReX::pyAMReX_${D}d ALIAS pyAMReX_${D}d)

    # own headers
    target_include_directories(pyAMReX_${D}d PUBLIC
        $<BUILD_INTERFACE:${pyAMReX_SOURCE_DIR}/src>
    )

    # if we include <AMReX_buildInfo.H> we will need to call:
    generate_buildinfo(pyAMReX_${D}d "${pyAMReX_SOURCE_DIR}")
    target_link_libraries(pyAMReX_${D}d PRIVATE buildInfo::pyAMReX_${D}d)
endforeach()

# add sources
add_subdirectory(src)

# set source language, properties, etc.
foreach(D IN LISTS AMReX_SPACEDIM)
    # link dependencies
    target_link_libraries(pyAMReX_${D}d PUBLIC AMReX::amrex_${D}d)
    target_link_libraries(pyAMReX_${D}d PRIVATE pybind11::module pybind11::windows_extras)
    if(pyAMReX_IPO)
        if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
            pyamrex_enable_IPO(pyAMReX_${D}d)
        else()
            # conditionally defined target in pybind11
            # https://github.com/pybind/pybind11/blob/v2.13.0/tools/pybind11Common.cmake#L407-L413
            target_link_libraries(pyAMReX_${D}d PRIVATE pybind11::lto)
        endif()
    endif()

    # set Python module properties
    set_target_properties(pyAMReX_${D}d PROPERTIES
        # hide symbols for combining multiple pybind11 modules downstream & for
        # reduced binary size
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        # name of the pybind-generated python module, which is wrapped in another
        # fluffy front-end modules, so we can extend it with pure Python
        ARCHIVE_OUTPUT_NAME amrex_${D}d_pybind
        LIBRARY_OUTPUT_NAME amrex_${D}d_pybind
        # build output directories - mainly set to run tests from CMake & IDEs
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
        PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
        COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
    )
    get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(isMultiConfig)
        foreach(CFG IN LISTS CMAKE_CONFIGURATION_TYPES)
            string(TOUPPER "${CFG}" CFG_UPPER)
            set_target_properties(pyAMReX_${D}d PROPERTIES
                # build output directories - mainly set to run tests from CMake & IDEs
                # note: same as above, but for Multi-Config generators
                ARCHIVE_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
                LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
                RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
                PDB_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
                COMPILE_PDB_OUTPUT_DIRECTORY_${CFG_UPPER} ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/space${D}d
            )
        endforeach()
    endif()
    if(EMSCRIPTEN)
        set_target_properties(pyAMReX_${D}d PROPERTIES
            PREFIX "")
    else()
        pybind11_extension(pyAMReX_${D}d)
    endif()
    if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
        pybind11_strip(pyAMReX_${D}d)
    endif()

    # C++ properties: at least a C++17 capable compiler is needed
    # AMReX helper function: propagate CUDA specific target & source properties
    if(AMReX_GPU_BACKEND STREQUAL CUDA)
        setup_target_for_cuda_compilation(pyAMReX_${D}d)
        target_compile_features(pyAMReX_${D}d PUBLIC cuda_std_17)
        set_target_properties(pyAMReX_${D}d PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    else()
        target_compile_features(pyAMReX_${D}d PUBLIC cxx_std_17)
        set_target_properties(pyAMReX_${D}d PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON
        )
    endif()
endforeach()


# Defines #####################################################################
#
# none needed - please use AMReX_Config.H


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
# these files are used if pyAMReX is installed and picked up by a downstream
# project
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${pyAMReX_SOURCE_DIR}/pyAMReXConfig.cmake.in
    ${pyAMReX_BINARY_DIR}/pyAMReXConfig.cmake
    INSTALL_DESTINATION ${pyAMReX_INSTALL_CMAKEDIR}
    #PATH_VARS MODULE_PATH
    # We have our own check_required_components
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file("pyAMReXConfigVersion.cmake"
    VERSION ${pyAMReX_VERSION}
    COMPATIBILITY SameMinorVersion
)


# Installs ####################################################################
#

# headers, libraries and executables
foreach(D IN LISTS AMReX_SPACEDIM)
    set(pyAMReX_INSTALL_TARGET_NAMES ${pyAMReX_INSTALL_TARGET_NAMES} pyAMReX_${D}d)
endforeach()

if(pyAMReX_INSTALL)
    install(TARGETS ${pyAMReX_INSTALL_TARGET_NAMES}
        EXPORT pyAMReXTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # CMake package file for find_package(pyAMReX) in depending projects
    install(EXPORT pyAMReXTargets
        FILE pyAMReXTargets.cmake
        NAMESPACE pyAMReX::
        DESTINATION ${pyAMReX_INSTALL_CMAKEDIR}
    )
    install(
        FILES
            ${pyAMReX_BINARY_DIR}/pyAMReXConfig.cmake
            ${pyAMReX_BINARY_DIR}/pyAMReXConfigVersion.cmake
        DESTINATION ${pyAMReX_INSTALL_CMAKEDIR}
    )
endif()


# pip helpers for the amrex package ###########################################
#
set(PY_PIP_OPTIONS "-v" CACHE STRING
    "Additional parameters to pass to `pip` as ; separated list")
set(PY_PIP_INSTALL_OPTIONS "" CACHE STRING
    "Additional parameters to pass to `pip install` as ; separated list")

# add a prefix to custom targets so we do not collide if used as a subproject
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_pyAMReX_CUSTOM_TARGET_PREFIX_DEFAULT "")
else()
    set(_pyAMReX_CUSTOM_TARGET_PREFIX_DEFAULT "pyamrex_")
endif()
set(pyAMReX_CUSTOM_TARGET_PREFIX "${_pyAMReX_CUSTOM_TARGET_PREFIX_DEFAULT}"
    CACHE STRING "Prefix for custom targets")

# build the wheel by re-using the shared library we build
add_custom_target(${pyAMReX_CUSTOM_TARGET_PREFIX}pip_wheel
    ${CMAKE_COMMAND} -E rm -f -r amrex-whl
    COMMAND
        ${CMAKE_COMMAND} -E env PYAMREX_LIBDIR=${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex/
            ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} wheel --no-build-isolation --no-deps --wheel-dir=amrex-whl "${pyAMReX_SOURCE_DIR}"
    COMMAND_EXPAND_LISTS VERBATIM
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
    DEPENDS
        ${pyAMReX_INSTALL_TARGET_NAMES}
)

# this will also upgrade/downgrade dependencies, e.g., when the version of numpy changes
add_custom_target(${pyAMReX_CUSTOM_TARGET_PREFIX}pip_install_requirements
    ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install ${PY_PIP_INSTALL_OPTIONS} -r "${pyAMReX_SOURCE_DIR}/requirements.txt"
    COMMAND_EXPAND_LISTS VERBATIM
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
)

# We force-install because in development, it is likely that the version of
# the package does not change, but the code did change. We need --no-deps,
# because otherwise pip would also force reinstall all dependencies.
add_custom_target(${pyAMReX_CUSTOM_TARGET_PREFIX}pip_install
    ${CMAKE_COMMAND} -E env AMREX_MPI=${AMReX_MPI}
        ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS} --find-links=amrex-whl amrex
    COMMAND_EXPAND_LISTS VERBATIM
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
    DEPENDS
        ${pyAMReX_INSTALL_TARGET_NAMES}
        ${pyAMReX_CUSTOM_TARGET_PREFIX}pip_wheel
        ${pyAMReX_CUSTOM_TARGET_PREFIX}pip_install_requirements
)

# this is for package managers only
add_custom_target(${pyAMReX_CUSTOM_TARGET_PREFIX}pip_install_nodeps
    ${CMAKE_COMMAND} -E env AMREX_MPI=${AMReX_MPI}
        ${Python_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS} --find-links=amrex-whl amrex
    COMMAND_EXPAND_LISTS VERBATIM
    WORKING_DIRECTORY
        ${pyAMReX_BINARY_DIR}
    DEPENDS
        ${pyAMReX_INSTALL_TARGET_NAMES}
        ${pyAMReX_CUSTOM_TARGET_PREFIX}pip_wheel
)


# Tests #######################################################################
#
if(pyAMReX_BUILD_TESTING)
    # copy Python wrapper library to build directory
    add_custom_command(TARGET pyAMReX_${AMReX_SPACEDIM_LAST}d POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${pyAMReX_SOURCE_DIR}/src/amrex
            $<TARGET_FILE_DIR:pyAMReX_${AMReX_SPACEDIM_LAST}d>/..
    )

    add_test(NAME pytest.AMReX
        COMMAND ${Python_EXECUTABLE} -m pytest -s -vvvv
            ${pyAMReX_SOURCE_DIR}/tests
        WORKING_DIRECTORY
            ${CMAKE_PYTHON_OUTPUT_DIRECTORY}
    )

    # limit threads
    set_property(TEST pytest.AMReX APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=3")

    # set PYTHONPATH and PATH (for .dll files)
    pyamrex_test_set_pythonpath(pytest.AMReX)
endif()


# Status Summary for Build Options ############################################
#
pyAMReX_print_summary()
