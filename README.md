# matrix_factorization

C++ code for non-negative matrix factorization joint optimization. Includes NMF, NMTF and Symmetric NMTF. 

To compile, gcc supporting C++11 (or above) and cmake are required.
Dependencies include Armadillo, Lapack, OpenBlas (or MKL), CLapack, HDF5, OpenMP(see CmakeLists.txt file for details). 

EDIT CMakeLists.txt TO BUILD LIBRARY.

This is a fully non-negative framework, i.e. all factors are constrained to be positive (For update rules see Čopar, Andrej, and Blaž Zupan. "Scalable non-negative matrix tri-factorization." BioData mining 10.1 (2017): 41.)

Globally, we attempt to write the code in an intuitive manner. See the example main for an illustration. The core idea is to first read in the data, then define the various factors. Based on the factors and data, one can then define the various objective functions. Once everything is define, the class Integration can be used to run the joint optimization of the different objective.
