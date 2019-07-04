cmake_minimum_required(VERSION 2.8)

project(run_framework CXX)

set(CMAKE_CXX_STANDARD 11)
# Replace PATH_TO_ARMADILLO_INCLUDE_FOLDER
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I PATH_TO_ARMADILLO_INCLUDE_FOLDER -DARMA_DONT_USE_WRAPPER")

# CHOOSE BETWEEN OPENBLAS OR MKL FOR FAST MATRIX OPERATIONS:
# If using OpenBlas uncoment line below and replace PATH_TO_OPENBLAS_LIB_FOLDER
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -LPATH_TO_OPENBLAS_LIB_FOLDER -lopenblas")

# If using MKL uncomment the two lines below
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -I${MKLROOT}/include ")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")

# Include Lapack, replace PATH_TO_LAPACK_LIB_FOLDER
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -LPATH_TO_LAPACK_LIB_FOLDER -llapack")
# Atlas and clapack are require if using the semi-constrained framework, replace PATH_TO_ATLAS_FOLDER_CONTANING_CLAPACK
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -LPATH_TO_ATLAS_FOLDER_CONTANING_CLAPACK -lclapack")

# Include HDF5 used for reading and saving data to binary format, replace PATH_TO_HDF5_LIB_FOLDER
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -LPATH_TO_HDF5_LIB_FOLDER -lhdf5")

# Generic compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -O3 -march=native -fopenmp")

add_executable(${PROJECT_NAME} "main.cpp" "source/factor.cpp" "source/sidefactor.cpp" "source/centralfactor.cpp" "source/objective.cpp" "source/nmfobjective.cpp" "source/nmtfobjective.cpp" "source/snmtfobjective.cpp" "source/integration.cpp" "source/crossvalidation.cpp")

