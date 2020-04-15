cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

project(nmfif CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake_aux/")

include(ARMA_FindMKL)
include(ARMA_FindOpenBLAS)
include(ARMA_FindLAPACK)

if(MKL_FOUND)
  set(LIBS ${LIBS} ${MKL_LIBRARIES})
else()
  if (OpenBLAS_FOUND)
    set(LIBS ${LIBS} ${OpenBLAS_LIBRARIES})
  else()
    message(FATAL_ERROR "MKL or OpenBLAS libraries not found")
  endif()
endif()

if(LAPACK_FOUND)
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES})
else()
  message(FATAL_ERROR "Lapack library not found")
endif()

# Replace PATH_TO_HDF5_LIB_FOLDER
find_library(HDF5_LIB hdf5  PATH_TO_HDF5_LIB_FOLDER)
if(NOT HDF5_LIB)
  message(FATAL_ERROR "hdf5 library not found")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -O3 -march=native -fopenmp -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5")

add_library(${PROJECT_NAME} SHARED "source/factor.cpp" "source/objective.cpp" "source/nmfobjective.cpp" "source/nmtfobjective.cpp" "source/snmtfobjective.cpp" "source/integration.cpp")
# Replace PATH_TO_HDF5_INCLUDE_FOLDER and PATH_TO_ARMADILLO_INCLUDE_FOLDER
target_include_directories(${PROJECT_NAME} PUBLIC PATH_TO_HDF5_INCLUDE_FOLDER PATH_TO_ARMADILLO_INCLUDE_FOLDER)
target_link_libraries(${PROJECT_NAME} PUBLIC ${HDF5_LIB} ${LIBS})


set(TEST "StaticTest")
add_executable(${TEST} "main.cpp")
target_link_libraries(${TEST} ${PROJECT_NAME})
