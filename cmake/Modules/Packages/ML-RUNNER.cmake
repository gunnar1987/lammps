enable_language(Fortran)
find_package(RUNNER QUIET)

if(RUNNER_FOUND)
  set(DOWNLOAD_RUNNER_DEFAULT OFF)
else()
  set(DOWNLOAD_RUNNER_DEFAULT ON)
endif()
option(DOWNLOAD_RUNNER "Download the RUNNER library instead of using an already installed one" ${DOWNLOAD_RUNNER_DEFAULT})
if(DOWNLOAD_RUNNER)
  # TODO actually download RuNNer and compile it to get library
  add_library(LAMMPS::RUNNER UNKNOWN IMPORTED)
  set_target_properties(LAMMPS::RUNNER PROPERTIES
    IMPORTED_LOCATION "librunner.a"
    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
  target_link_libraries(lammps PRIVATE LAMMPS::RUNNER)
else()
  find_package(RUNNER REQUIRED)
  target_link_libraries(lammps PRIVATE LAMMPS::RUNNER ${LAPACK_LIBRARIES})
endif()
