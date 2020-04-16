# NMFIF

**READ DOCUMENTATION FOR FULL DETAILS**

C++ code for non-negative matrix factorization joint optimization. Includes NMF, Symmetric NMF, NMTF and Symmetric NMTF. 

To compile, gcc supporting C++11 (or above) and cmake v2.8 (or above) are required.
Dependencies include Armadillo, Lapack, OpenBlas (or MKL), HDF5, OpenMP(see CmakeLists.txt file for details). 

**EDIT CMakeLists.txt TO BUILD THE LIBRARY ADDING THE PATH TO YOUR DOWNLOAD/INSTALLATION OF ARMADILLO AND HDF5.** Once edited, run the following commands to build the shared object.
```
$ cmake .
$ make
```

This is a fully non-negative framework, i.e. all factors are constrained to be positive.

Globally, we attempt to write the code in an intuitive manner. See the example main for an illustration. The core idea is to first read in the data, then define the various factors. Based on the factors and data, one can then define the various objective functions. Once everything is defined, the class Integration can be used to run the joint optimization of the different objectives.

The **examples** folder contains example main.cpp files with CMakeLists.txt files to compile them (the CMakeLists.txt need to be edited!!!)
