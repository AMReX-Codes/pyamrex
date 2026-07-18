function(find_nanobind)
    if(COMMAND nanobind_add_module)
        message(STATUS "nanobind CMake helpers already imported")
    elseif(pyAMReX_nanobind_src)
        message(STATUS "Compiling local nanobind ...")
        message(STATUS "nanobind source path: ${pyAMReX_nanobind_src}")
        if(NOT IS_DIRECTORY ${pyAMReX_nanobind_src})
            message(FATAL_ERROR "Specified directory pyAMReX_nanobind_src='${pyAMReX_nanobind_src}' does not exist!")
        endif()
    elseif(pyAMReX_nanobind_internal)
        message(STATUS "Downloading nanobind ...")
        message(STATUS "nanobind repository: ${pyAMReX_nanobind_repo} (${pyAMReX_nanobind_branch})")
        include(FetchContent)
    endif()

    if(COMMAND nanobind_add_module)
        # The helpers were provided by a parent project.
    elseif(pyAMReX_nanobind_internal OR pyAMReX_nanobind_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
        if(pyAMReX_nanobind_src)
            add_subdirectory(${pyAMReX_nanobind_src} _deps/localnanobind-build/)
        else()
            FetchContent_Declare(fetchednanobind
                GIT_REPOSITORY ${pyAMReX_nanobind_repo}
                GIT_TAG        ${pyAMReX_nanobind_branch}
                GIT_SUBMODULES_RECURSE TRUE
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchednanobind)

            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDnanobind)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDnanobind)
        endif()
    else()
        find_package(nanobind ${nanobind_version_min} CONFIG REQUIRED)
        message(STATUS "nanobind: Found version '${nanobind_VERSION}'")
    endif()
endfunction()

set(pyAMReX_nanobind_src ""
    CACHE PATH
    "Local path to nanobind source directory (preferred if set)")

option(pyAMReX_nanobind_internal "Download & build nanobind" ON)
set(pyAMReX_nanobind_repo "https://github.com/wjakob/nanobind.git"
    CACHE STRING
    "Repository URI to pull and build nanobind from if(pyAMReX_nanobind_internal)")

file(READ "${pyAMReX_SOURCE_DIR}/dependencies.json" dependencies_data)
string(JSON nanobind_version_min GET "${dependencies_data}" version_nanobind_min)
string(JSON nanobind_commit GET "${dependencies_data}" commit_nanobind)
string(REGEX REPLACE "^v" "" nanobind_version_min "${nanobind_version_min}")

set(pyAMReX_nanobind_branch ${nanobind_commit}
    CACHE STRING
    "Repository branch for pyAMReX_nanobind_repo if(pyAMReX_nanobind_internal)")

find_nanobind()
