if (NOT AMReX_EB)
    return()
endif()

foreach(D IN LISTS AMReX_SPACEDIM)
    if (D GREATER 1)
        target_sources(pyAMReX_${D}d
          PRIVATE
            EB.cpp
            EBFabFactory.cpp
        )
    endif()
endforeach()
