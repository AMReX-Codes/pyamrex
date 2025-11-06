foreach(D IN LISTS AMReX_SPACEDIM)
    if (D EQUAL 1)
        return()
    endif()

    target_sources(pyAMReX_${D}d
      PRIVATE
        EB.cpp
        EBFabFactory.cpp
    )
endforeach()
