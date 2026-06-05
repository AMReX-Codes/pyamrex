foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        AmrCore.cpp
        AmrMesh.cpp
    )
endforeach()
