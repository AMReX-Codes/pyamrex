function(find_nanobind)
    if(TARGET nanobind::module)
        message(STATUS "nanobind::module target already imported")
    elseif(pyAMReX_nanobind_src)
        message(STATUS "Compiling local nanobind ...")
        message(STATUS "nanobind source path: ${pyAMReX_nanobind_src}")
    elseif(pyAMReX_nanobind_internal)
        message(STATUS "Downloading nanobind ...")
        message(STATUS "nanobind repository: ${pyAMReX_nanobind_repo} (${pyAMReX_nanobind_branch})")
        include(FetchContent)
    endif()
    if(TARGET nanobind::module)
        # nothing to do, target already exists in the superbuild
    elseif(pyAMReX_nanobind_internal OR pyAMReX_nanobind_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        if(pyAMReX_nanobind_src)
            add_subdirectory(${pyAMReX_nanobind_src} _deps/localnanobind-build/)
        else()
            FetchContent_Declare(fetchednanobind
                GIT_REPOSITORY ${pyAMReX_nanobind_repo}
                GIT_TAG        ${pyAMReX_nanobind_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_GetProperties(fetchednanobind)

            if(NOT fetchednanobind_POPULATED)
                FetchContent_Populate(fetchednanobind)
                add_subdirectory(${fetchednanobind_SOURCE_DIR} ${fetchednanobind_BINARY_DIR})
            endif()

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDnanobind)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDnanobind)
        endif()
    elseif(NOT pyAMReX_nanobind_internal)
        find_package(nanobind 1.2.0 CONFIG REQUIRED)
        message(STATUS "nanobind: Found version '${nanobind_VERSION}'")
    endif()
endfunction()

# local source-tree
set(pyAMReX_nanobind_src ""
    CACHE PATH
    "Local path to nanobind source directory (preferred if set)")

# Git fetcher
option(pyAMReX_nanobind_internal "Download & build nanobind" ON)
set(pyAMReX_nanobind_repo "https://github.com/wjakob/nanobind.git"
    CACHE STRING
    "Repository URI to pull and build nanobind from if(pyAMReX_nanobind_internal)")
set(pyAMReX_nanobind_branch "v1.2.0"
    CACHE STRING
    "Repository branch for pyAMReX_nanobind_repo if(pyAMReX_nanobind_internal)")

find_nanobind()
