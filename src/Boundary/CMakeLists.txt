foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        BCRec.cpp
        BCUtil.cpp
        PhysBCFunct.cpp
    )
endforeach()
