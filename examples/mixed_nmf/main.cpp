#include <iostream>
#include "integration.h"


// EXAMPLE MAIN, we have two main matrices R12 and R23 that we simultaneously factorize using a combination of NMF and NMTF
int main()
{
    //Uncomment if needed
    //omp_set_dynamic(0);
    //omp_set_num_threads(8);

    std::cout << "Hello World!" << std::endl;

    double start = omp_get_wtime();
    // Reading files in, mind that C++ assumes column-major format which does not always correspond to how the data was generated.
    // Check that the matrix read is as you expect, the option arma::hdf5_opts::trans perform transposition if it isn't
    arma::mat X; X.load(arma::hdf5_name("../data/test_204_948","dataset",arma::hdf5_opts::trans));
    arma::mat Y; Y.load(arma::hdf5_name("../data/test_694_204","dataset",arma::hdf5_opts::trans));
    arma::mat Z; Z.load(arma::hdf5_name("../data/test_204_204","dataset",arma::hdf5_opts::trans));
    std::cout << "Data read in " << (double)omp_get_wtime() - start << std::endl;

    unsigned int n1 = arma::size(X)(0);
    unsigned int n2 = arma::size(X)(1);
    unsigned int n3 = arma::size(Y)(0);
    unsigned int k1 = 20;
    unsigned int k2 = 50;
    unsigned int k3 = 73;

    std::string initializer = "random";

    factors all_factors;
    // We first create all the factors and add them to a list
    factor F("./",n1,k1,initializer); all_factors.push_back(&F);
    factor S1("./",k1,k2,initializer); all_factors.push_back(&S1);
    factor G("./",n2,k2,initializer); G.addL2Regularizer(); all_factors.push_back(&G);
    factor S2("./",k3,k1,initializer); all_factors.push_back(&S2);
    factor H("./",n3,k3,initializer); all_factors.push_back(&H);

    std::cout << "Factors initialized in " << (double)omp_get_wtime() - start << std::endl;

    objectives all_objectives;
    // We then initialize the different objectives and add them to a list
    nmtfObjective O1(X.memptr(),arma::size(X),&F,&S1,&G); all_objectives.push_back(&O1);
    nmtfObjective O2(Y.memptr(),arma::size(Y),&H,&S2,&F); all_objectives.push_back(&O2);
    snmfObjective O3(Z.memptr(),arma::size(Z),&F); all_objectives.push_back(&O3);

    std::cout << "Objectives initialized in " << (double)omp_get_wtime() - start << std::endl;

    // Finally we create an integration object with all factors and objectives
    integration test(all_objectives, all_factors);

    test.optimize(100,10e-6,"./losses.txt");

    return 0;
}
