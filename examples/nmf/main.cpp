#include <iostream>
#include "integration.h"


int main()
{
    //Uncomment if needed
    //omp_set_dynamic(0);
    //omp_set_num_threads(8);

    std::cout << "Hello World!" << std::endl;

    double start = omp_get_wtime();
    // Reading files in, mind that C++ assumes column-major format which does not always correspond to how the data was generated.
    // Check that the matrix read is as you expect, the option arma::hdf5_opts::trans perform transposition if it isn't
    arma::mat X; X.load(arma::hdf5_name("../data/test_333_948","dataset",arma::hdf5_opts::trans));
    std::cout << "Data read in " << (double)omp_get_wtime() - start << std::endl;

    unsigned int n1 = arma::size(X)(0);
    unsigned int n2 = arma::size(X)(1);
    unsigned int k = 20;

    std::string initializer = "random";

    factors all_factors;
    // We first create all the factors and add them to a list
    factor F("./",n1,k,initializer); all_factors.push_back(&F);
    factor G("./",n2,k,initializer); all_factors.push_back(&G);

    std::cout << "Factors initialized in " << (double)omp_get_wtime() - start << std::endl;

    objectives all_objectives;
    // We then initialize the different objectives and add them to a list
    nmfObjective O(X.memptr(),arma::size(X),&F,&G); all_objectives.push_back(&O);

    std::cout << "Objectives initialized in " << (double)omp_get_wtime() - start << std::endl;

    // Finally we create an integration object with all factors and objectives
    integration test(all_objectives, all_factors);

    test.optimize(100,10e-6,"./losses.txt");

    return 0;
}
