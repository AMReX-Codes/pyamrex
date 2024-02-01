AMREX_HOME = /home/bsiddani/amrex

DEBUG	= FALSE
DEBUG	= TRUE

DIM	= 3

COMP    = gcc

USE_MPI   = TRUE
USE_AMREX_MPMD = TRUE

USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(USE_AMREX_MPMD), TRUE)
  DEFINES += -DUSE_AMREX_MPMD
endif
