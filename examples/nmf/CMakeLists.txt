cmake_minimum_required(VERSION 2.8)

project(test_nmf CXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -fopenmp -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5")
add_executable(${PROJECT_NAME} "main.cpp")


find_library(NMFIF_LIB nmfif  PATH_TO_FOLDER_CONTAINING_NMFIF_SHARED_OBJECT)
if(NOT NMFIF_LIB)
  message(FATAL_ERROR "nmfif library not found")
endif()
find_library(HDF5_LIB hdf5 PATH_TO_HDF5_LIB_FOLDER)
if(NOT HDF5_LIB)
  message(FATAL_ERROR "hdf5 library not found")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${NMFIF_LIB} ${HDF5_LIB})
target_include_directories(${PROJECT_NAME} PUBLIC PATH_TO_NMFIF_SOURCE_FOLDER@ PATH_TO_HDF5_INCLUDE_FOLDER@ PATH_TO_ARMADILLO_INCLUDE_FOLDER)
