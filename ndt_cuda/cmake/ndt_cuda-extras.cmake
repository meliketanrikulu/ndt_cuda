# Ubuntu 24.04 (ROS 2 Jazzy): the VTK 9.1 package that PCL pulls in references the MPI::MPI_C target. When MPI is
# not installed that target is undefined and every project that finds PCL fails to configure, although VTK's MPI
# module is not used. Provide an empty placeholder in that case. Included by this project and, before PCL is
# searched, by the package configuration installed for consumers.
if(NOT TARGET MPI::MPI_C)
  find_package(MPI QUIET COMPONENTS C)
endif()
if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
endif()
