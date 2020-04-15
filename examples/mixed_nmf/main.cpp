#include <iostream>
#include "source/integration.h"


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
    arma::mat R12; R12.load(arma::hdf5_name("path_to_hdf5_format_file","dataset",arma::hdf5_opts::trans));
    arma::mat R23; R23.load(arma::hdf5_name("path_to_hdf5_format_file","dataset",arma::hdf5_opts::trans));
    arma::mat regularizer; regularizer.load(arma::hdf5_name("path_to_hdf5_format_file","dataset",arma::hdf5_opts::trans));
    std::cout << "Data read in " << (double)omp_get_wtime() - start << std::endl;

    unsigned int n1 = arma::size(R12)(0);
    unsigned int n2 = arma::size(R12)(1);
    unsigned int n3 = arma::size(R23)(1);

    unsigned int k1 = 20;
    unsigned int k2 = 20;
    std::string initializer = "nnsvd";
    // We simultaneously factorize R12 = G1 * S * G2.T and R23 = G2 * G3.T

    factors all_factors;
    // We first create all the factors and add them to a list
    factor G1("path_for_save",n1,k1,initializer); G1.addRegularizer(regularizer.memptr(),arma::size(regularizer)); all_factors.push_back(&G1);
    factor G2("path_for_save",n2,k2,initializer); all_factors.push_back(&G2);
    factor G3("path_for_save",n3,k2,initializer);  all_factors.push_back(&G3);

    factor S("path_for_save",k1,k2); all_factors.push_back(&S);

    std::cout << "Factors initialized in " << (double)omp_get_wtime() - start << std::endl;

    objectives all_objectives;
    // We then initialize the different objectives and add them to a list
    nmtfObjective O12(R12.memptr(),arma::size(R12),&G1,&S,&G2); all_objectives.push_back(&O12); // R12 = G1 * S * G2.T
    nmfObjective O23(R23.memptr(),arma::size(R23),&G2,&G3); all_objectives.push_back(&O23);// R23 = G2 * G3.T

    std::cout << "Objectives initialized in " << (double)omp_get_wtime() - start << std::endl;

    // Finally we create an integration object with all factors and objectives
    integration test(all_objectives, all_factors);
    unsigned int total_number_of_iterations,number_of_iterations_between_loss_updates;
    std::string path_to_file_where_to_output_losses;

    test.optimize(total_number_of_iterations,number_of_iterations_between_loss_updates,path_to_file_where_to_output_losses);

    return 0;
}
