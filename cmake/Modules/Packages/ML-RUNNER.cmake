enable_language(Fortran)
find_package(RUNNER QUIET)

if(RUNNER_FOUND)
  set(DOWNLOAD_RUNNER_DEFAULT OFF)
else()
  set(DOWNLOAD_RUNNER_DEFAULT ON)
endif()
option(DOWNLOAD_RUNNER "Download the RUNNER library instead of using an already installed one" ${DOWNLOAD_RUNNER_DEFAULT})
if(DOWNLOAD_RUNNER)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" BTYPE)
  set(temp "F77 = ${CMAKE_Fortran_COMPILER}\nF90 = ${CMAKE_Fortran_COMPILER}\nF95 = ${CMAKE_Fortran_COMPILER}\n")
  set(temp "${temp}CC=${CMAKE_C_COMPILER}\nCPLUSPLUS=${CMAKE_CXX_COMPILER}\nLINKER=${CMAKE_Fortran_COMPILER}\n")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
    set(temp "${temp}FPP=${CMAKE_Fortran_COMPILER} -E\nOPTIM=${CMAKE_Fortran_FLAGS_${BTYPE}}\n")
    set(temp "${temp}DEFINES += -DGETARG_F2003 -DFORTRAN_UNDERSCORE\n")
    set(temp "${temp}F95FLAGS += -fpp -free -fPIC\n")
    set(temp "${temp}F77FLAGS += -fpp -fixed -fPIC\n")
  elseif(CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
    set(temp "${temp}FPP=${CMAKE_Fortran_COMPILER} -E -x f95-cpp-input\nOPTIM=${CMAKE_Fortran_FLAGS_${BTYPE}}\n")
    set(temp "${temp}DEFINES += -DGETARG_F2003 -DGETENV_F2003 -DGFORTRAN -DFORTRAN_UNDERSCORE\n")
    set(temp "${temp}F95FLAGS += -x f95-cpp-input -ffree-line-length-none -ffree-form -fno-second-underscore -fPIC\n")
    set(temp "${temp}F77FLAGS += -x f77-cpp-input -fno-second-underscore -fPIC\n")
  else()
    message(FATAL_ERROR "The ${CMAKE_Fortran_COMPILER_ID} Fortran compiler is not (yet) supported for building RUNNER")
  endif()
  set(temp "${temp}CFLAGS += -fPIC \nCPLUSPLUSFLAGS += -fPIC\nAR_ADD=src\n")
  set(temp "${temp}MATH_LINKOPTS=")
  foreach(flag ${BLAS_LIBRARIES})
    set(temp "${temp} ${flag}")
  endforeach()
  foreach(flag ${LAPACK_LIBRARIES})
    set(temp "${temp} ${flag}")
  endforeach()
  # Fix cmake crashing when MATH_LINKOPTS not set, required for e.g. recent Cray Programming Environment
  set(temp "${temp} -L/_DUMMY_PATH_\n")
  set(temp "${temp}PYTHON=python\nPIP=pip\nEXTRA_LINKOPTS=\n")
  set(temp "${temp}HAVE_CP2K=0\nHAVE_VASP=0\nHAVE_TB=0\nHAVE_PRECON=1\nHAVE_LOTF=0\nHAVE_ONIOM=0\n")
  set(temp "${temp}HAVE_LOCAL_E_MIX=0\nHAVE_QC=0\nHAVE_GAP=1\nHAVE_DESCRIPTORS_NONCOMMERCIAL=1\n")
  set(temp "${temp}HAVE_TURBOGAP=0\nHAVE_QR=1\nHAVE_THIRDPARTY=0\nHAVE_FX=0\nHAVE_SCME=0\nHAVE_MTP=0\n")
  set(temp "${temp}HAVE_MBD=0\nHAVE_TTM_NF=0\nHAVE_CH4=0\nHAVE_NETCDF4=0\nHAVE_MDCORE=0\nHAVE_ASAP=0\n")
  set(temp "${temp}HAVE_CGAL=0\nHAVE_METIS=0\nHAVE_LMTO_TBE=0\nHAVE_SCALAPACK=0\n")
  file(WRITE ${CMAKE_BINARY_DIR}/runner2.config "${temp}")

  message(STATUS "RUNNER download via git requested - we will build our own")
  set(CMAKE_EP_GIT_REMOTE_UPDATE_STRATEGY CHECKOUT)
  # RUNNER has no releases (except for a tag marking the end of Python 2 support). We use the current "public" branch
  # The LAMMPS interface wrapper has a compatibility constant that is being checked at runtime.
  include(ExternalProject)
  ExternalProject_Add(runner2_build
    GIT_REPOSITORY "https://github.com/libAtoms/QUIP/"
    GIT_TAG origin/public
    GIT_SHALLOW YES
    GIT_PROGRESS YES
    GIT_SUBMODULES "src/fox;src/GAP"
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/runner2.config <SOURCE_DIR>/arch/Makefile.lammps
    CONFIGURE_COMMAND env QUIP_ARCH=lammps make config
    BUILD_COMMAND env QUIP_ARCH=lammps make libquip
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE YES
    BUILD_BYPRODUCTS <SOURCE_DIR>/build/lammps/libquip.a
  )
  ExternalProject_get_property(runner2_build SOURCE_DIR)
  add_library(LAMMPS::RUNNER UNKNOWN IMPORTED)
  set_target_properties(LAMMPS::RUNNER PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/build/lammps/libquip.a"
    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
  target_link_libraries(lammps PRIVATE LAMMPS::RUNNER)
  add_dependencies(LAMMPS::RUNNER runner2_build)
else()
  find_package(RUNNER REQUIRED)
  target_link_libraries(lammps PRIVATE RUNNER::RUNNER ${LAPACK_LIBRARIES})
endif()
