#include "integration.h"
#include <algorithm>
#include <random>

void integration::optimize(const unsigned int max_iter, double threshold, const std::string &filename)
{
    for (factors::reverse_iterator F=Fs.rbegin(); F!=Fs.rend();++F) {
        (*F)->initialiseWeights();
    }

    arma::mat storeLosses(numObjectives,(int)(max_iter),arma::fill::zeros);
    for (unsigned int i=0; i<max_iter; ++i){
        float start = omp_get_wtime();
        std::random_shuffle(Fs.begin(),Fs.end());
        for (factors::reverse_iterator F=Fs.rbegin(); F!=Fs.rend();++F) {
            (*F)->step();
        }
        std::cout << "Losses:";
        int j=0;
        for (objectives::iterator O=Os.begin(); O!=Os.end(); ++O) {
            (*O)->computeLoss();
            storeLosses(j,i) = (*O)->getLoss();
            std::cout << " " << storeLosses(j,i);
            ++j;
        std::cout << "\n";
        }
        if (i > 0){
            double jn = arma::accu(storeLosses(arma::span::all,i-1));
            double delta = (jn-arma::accu(storeLosses(arma::span::all,i)))/jn;
            if (delta < threshold && delta > 0)  {
                std::cout << "Early stop at iteration " << i << " with delta " << delta << std::endl;
                break;
            }

            std::cout << "\tIteration "<< i+1 <<"/"<< max_iter << " completed in "<< (float)omp_get_wtime() - start<< " seconds. (delta ="<<delta<<")" << std::endl;
        }
    }

    storeLosses.save(arma::hdf5_name(filename,"dataset",arma::hdf5_opts::trans));
}


double integration::computeLoss()
{
    double score = 0.0;
    for (objectives::iterator O=Os.begin(); O!=Os.end(); ++O) {
        (*O)->computeLoss();
        score += (*O)->getLoss();
    }
    return score;
}
