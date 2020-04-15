cmake_minimum_required(VERSION 2.8)

project(test_nmf CXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -fopenmp -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5")
add_executable(${PROJECT_NAME} "main.cpp")


find_library(NMFIF_LIB nmfif ../../)
if(NOT NMFIF_LIB)
  message(FATAL_ERROR "nmfif library not found")
endif()
find_library(HDF5_LIB hdf5 /home/thomas/Documents/hdf5/lib)
if(NOT HDF5_LIB)
  message(FATAL_ERROR "hdf5 library not found")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${NMFIF_LIB} ${HDF5_LIB})
target_include_directories(${PROJECT_NAME} PUBLIC /home/thomas/Documents/phd_code/matrix_factorization/source /home/thomas/Documents/hdf5/include /home/thomas/Documents/armadillo-9.860.2/include)
