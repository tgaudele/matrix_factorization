# matrix_factorization

C++ framework for Non-negative matrix factorization integration. Includes NMF, NMTF and Symmetric NMTF.

To compile, gcc supporting C++11 (or above) and cmake are required.
Dependencies include Armadillo, Lapack, OpenBlas (or MKL), CLapack, HDF5, OpenMP(see CmakeLists.txt file for details).

There are two subframework not compatible due to difference in multiplicative update rules. They share the virtual classes Factor and Objective as well as the class Integration.

The first one corresponds to fully non-negative framework (all factors are constrained to be positive). The relevant classes for this framework are sideFactor.*, centralFactor.*, and all objective classes that do not contain the prefix "semi". (For update rules see Čopar, Andrej, and Blaž Zupan. "Scalable non-negative matrix tri-factorization." BioData mining 10.1 (2017): 41.)

The second framework corresponds to semi constrained in which one matrices (the central one for tri-factorization and the left one for bi-factorization) is unconstrained. The relevant classes for this framework are positiveFactor, unconstrainedFactor, and all objective classes that contain the prefix "semi". (For update rules see Wang, Fei, Tao Li, and Changshui Zhang. "Semi-supervised clustering via matrix factorization." Proceedings of the 2008 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2008.)

Globally, we attempt to write the code in an intuitive manner. See the example main for an illustration. The core idea is to first read in the data, then define the various factors. Based on the factors and data, one can then define the various objective functions. Once everything is define, the class Integration can be used to run the joint optimization of the different objective.
