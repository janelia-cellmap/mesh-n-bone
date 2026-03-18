# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.12...3.29)

project( cgal_skeletonize_mesh )


# CGAL and its components

# Point at your Conda environment first
list(INSERT CMAKE_PREFIX_PATH 0 "$ENV{CONDA_PREFIX}")
set(CGAL_USE_PKGCONFIG OFF CACHE BOOL "" FORCE)

# Find CGAL
find_package(CGAL 6.0 REQUIRED COMPONENTS Core)


if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()

endif()

# Eigen3 integration
find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
include(CGAL_Eigen3_support)   # Defines CGAL::Eigen3_support


# include for local directory

# include for local package


# Creating entries for all C++ files with "main" routine
# ##########################################################


create_single_source_cgal_program( "skeletonize_mesh.cpp" )
# Link in Eigen support
target_link_libraries(skeletonize_mesh
  PRIVATE
    CGAL::Eigen3_support
)


