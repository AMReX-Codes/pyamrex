foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        AmrMesh.cpp
    )
endforeach()
