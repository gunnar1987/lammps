enable_language(C)

find_package(HDF5 REQUIRED)
target_include_directories(lammps PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(lammps PRIVATE ${HDF5_LIBRARIES})