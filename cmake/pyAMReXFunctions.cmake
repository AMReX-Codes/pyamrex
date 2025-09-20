# Set C++17 for the whole build if not otherwise requested
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
#
macro(set_cxx17_superbuild)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CUDA_EXTENSIONS)
        set(CMAKE_CUDA_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endmacro()


# find the CCache tool and use it if found
#
macro(set_ccache)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        if(AMReX_GPU_BACKEND STREQUAL CUDA)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        endif()
    endif()
    mark_as_advanced(CCACHE_PROGRAM)
endmacro()


# set names and paths of temporary build directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(pyamrex_set_default_build_dirs)
    if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for archives")
        mark_as_advanced(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for libraries")
        mark_as_advanced(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                CACHE PATH "Build directory for binaries")
        mark_as_advanced(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_PYTHON_OUTPUT_DIRECTORY)
        set(CMAKE_PYTHON_OUTPUT_DIRECTORY
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/site-packages"
            CACHE PATH "Build directory for python modules"
        )
    endif()
endmacro()


# set names and paths of install directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(pyamrex_set_default_install_dirs)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        include(GNUInstallDirs)
        if(NOT CMAKE_INSTALL_CMAKEDIR)
            set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake"
                CACHE PATH "CMake config package location for installed targets")
            if(WIN32)
                set(CMAKE_INSTALL_LIBDIR Lib
                    CACHE PATH "Object code libraries")
                set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
            endif()
            mark_as_advanced(CMAKE_INSTALL_CMAKEDIR)
        endif()
    endif()

    if(WIN32)
        set(pyAMReX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}")
    else()
        set(pyAMReX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}/pyAMReX")
    endif()
endmacro()


# set names and paths for Python modules
# this needs to be slightly delayed until we found Python and know its
# major and minor version number
#
macro(pyamrex_set_default_install_dirs_python)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        # Python install and build output dirs
        if(NOT CMAKE_INSTALL_PYTHONDIR)
            if(WIN32)
                set(CMAKE_INSTALL_PYTHONDIR_DEFAULT
                    "${CMAKE_INSTALL_LIBDIR}/site-packages"
                )
            else()
                set(CMAKE_INSTALL_PYTHONDIR_DEFAULT
                    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
                )
            endif()
            set(CMAKE_INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR_DEFAULT}"
                CACHE STRING "Location for installed python package"
            )
        endif()
    endif()
endmacro()


# function to set the PYTHONPATH and PATH (for .dll files) on a test
# this avoids that we need to install our python packages to run ctest
#
function(pyamrex_test_set_pythonpath test_name)
    if(WIN32)
        string(REPLACE ";" "\\;" WIN_PYTHONPATH "$ENV{PYTHONPATH}")
        string(REPLACE ";" "\\;" WIN_PATH "$ENV{PATH}")  # DLLs
        string(REGEX REPLACE "/" "\\\\" WIN_PYTHON_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY})
        # shared library note:
        #   For Windows Python 3.8+, this also needs to be injected via
        #   os.add_dll_directory.
        #   https://github.com/python/cpython/issues/80266
        #   https://docs.python.org/3.8/library/os.html#os.add_dll_directory
        set_property(TEST ${test_name}
            APPEND PROPERTY ENVIRONMENT
                "PYTHONPATH=${WIN_PYTHON_OUTPUT_DIRECTORY}\;${WIN_PYTHONPATH}"
        )
        foreach(D IN LISTS AMReX_SPACEDIM)
            set_property(TEST ${test_name}
                APPEND PROPERTY ENVIRONMENT
                    "PATH=$<TARGET_FILE_DIR:pyAMReX_${D}d>\;$<TARGET_FILE_DIR:AMReX::amrex_${D}d>\;${WIN_PATH}"
            )
        endforeach()
    else()
        set_property(TEST ${test_name}
            APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_PYTHON_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH}"
        )
    endif()
endfunction()


# change the default CMAKE_BUILD_TYPE
# the default in CMake is Debug for historic reasons
#
macro(set_default_build_type default_build_type)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE ${default_build_type}
                CACHE STRING
                "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
            set_property(CACHE CMAKE_BUILD_TYPE
                PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
        endif()
        if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
            message(WARNING "CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}' is not one of "
                    "${CMAKE_CONFIGURATION_TYPES}. Is this a typo?")
        endif()
    endif()
endmacro()

# Testing logic with possibility to overwrite on a project basis in superbuilds
#
macro(set_testing_option project_build_var)
    include(CTest)
    mark_as_advanced(BUILD_TESTING) # automatically defined, default: ON
    if(DEFINED BUILD_TESTING)
        set(${project_build_var}_DEFAULT ${BUILD_TESTING})
    else()
        set(${project_build_var}_DEFAULT ON)
    endif()
    option(${project_build_var} "Build the pyAMReX tests"
           ${${project_build_var}_DEFAULT})
endmacro()

# Set compile warnings
#
function(pyamrex_set_compile_warnings tgt)
    # On Windows, Clang -Wall aliases -Weverything; default is /W3
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT WIN32)
        # list(APPEND CMAKE_CXX_FLAGS "-fsanitize=address") # address, memory, undefined
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
        # set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")
        # note: might still need a
        #   export LD_PRELOAD=libclang_rt.asan.so
        # or on Debian 9 with Clang 6.0
        #   export LD_PRELOAD=/usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.asan-x86_64.so:
        #                     /usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.ubsan_minimal-x86_64.so
        # at runtime when used with symbol-hidden code (e.g. pybind11 module)
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wshadow -Woverloaded-virtual -Wunreachable-code -Wno-array-bounds)
        if(NOT AMReX_GPU_BACKEND STREQUAL CUDA)
            # In older NVCC, -Wpedantic causes "warning: style of line directive is a GCC extension"
            target_compile_options(${tgt} PRIVATE -Wpedantic)
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # Warning C4503: "decorated name length exceeded, name was truncated"
        # Symbols longer than 4096 chars are truncated (and hashed instead)
        # Yes, you should build against the same C++ runtime and with same
        # configuration (Debug/Release). MSVC does inconvenient choices for their
        # developers, so be it. (Our Windows-users use conda-forge builds, which
        # are consistent.)
        target_compile_options(${tgt} PRIVATE /wd4503 /wd4251)
    endif ()
endfunction()


# Enables interprocedural optimization for a list of targets
#
function(pyamrex_enable_IPO all_targets_list)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT is_IPO_available)
    if(is_IPO_available)
        foreach(tgt IN ITEMS ${all_targets_list})
            set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        endforeach()
    else()
        message(FATAL_ERROR "Interprocedural optimization is not available, set pyAMReX_IPO=OFF")
    endif()
endfunction()


# Set an MPI_TEST_EXE variable for test runs which runs num_ranks
# ranks. On some systems, you might need to use the a specific
# mpiexec wrapper, e.g. on Summit (ORNL) pass the hint
# -DMPIEXEC_EXECUTABLE=$(which jsrun) to run ctest.
#
function(configure_mpiexec num_ranks)
    # OpenMPI root guard: https://github.com/open-mpi/ompi/issues/4451
    if("$ENV{USER}" STREQUAL "root")
        # calling even --help as root will abort and warn on stderr
        execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --help
            ERROR_VARIABLE MPIEXEC_HELP_TEXT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(${MPIEXEC_HELP_TEXT} MATCHES "^.*allow-run-as-root.*$")
                set(MPI_ALLOW_ROOT --allow-run-as-root)
            endif()
    endif()
    set(MPI_TEST_EXE
        ${MPIEXEC_EXECUTABLE}
        ${MPI_ALLOW_ROOT}
        ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
        PARENT_SCOPE
    )
endfunction()


# Prints a summary of pyAMReX options at the end of the CMake configuration
#
function(pyAMReX_print_summary)
    message("")
    message("pyAMReX build configuration:")
    message("  Version: ${pyAMReX_VERSION}")
    message("  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} "
                            "${CMAKE_CXX_COMPILER_VERSION} "
                            "${CMAKE_CXX_COMPILER_WRAPPER}")
    message("    ${CMAKE_CXX_COMPILER}")
    message("")
    message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
    message("        bin: ${CMAKE_INSTALL_BINDIR}")
    message("        lib: ${CMAKE_INSTALL_LIBDIR}")
    message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
    message("      cmake: ${pyAMReX_INSTALL_CMAKEDIR}")
    message("     python: ${CMAKE_INSTALL_PYTHONDIR}")
    message("")
    message("  Build type: ${CMAKE_BUILD_TYPE}")
    #message("  Build options:")
    #message("    ABC: ${pyAMReX_ABC}")
    message("")
endfunction()
