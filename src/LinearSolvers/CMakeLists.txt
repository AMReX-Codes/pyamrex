foreach(D IN LISTS AMReX_SPACEDIM)
    target_sources(pyAMReX_${D}d
      PRIVATE
        MLLinOp.cpp
        MLABecLaplacian.cpp
        MLMG.cpp
        MLNodeLaplacian.cpp
        MLPoisson.cpp
    )
endforeach()
