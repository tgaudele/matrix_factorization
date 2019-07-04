#include "integration.h"

void integration::reset()
{
    for (factors::iterator F=Fs.begin();F!=Fs.end();++F){
        (*F)->reset();
    }
}


void integration::optimize(const unsigned int n_iter, const unsigned int n_update, const std::string &filename)
{
    arma::mat storeLosses(numObjectives,(int)(n_iter/n_update),arma::fill::zeros);
    for (unsigned int i=0; i<n_iter; ++i){
        float start = omp_get_wtime();
        for (factors::reverse_iterator F=Fs.rbegin(); F!=Fs.rend();++F) {
            (*F)->step(0.0);
        }
        if (i % n_update == 0) {
            std::cout << "Losses:";
            int j=0;
            for (objectives::iterator O=Os.begin(); O!=Os.end(); ++O) {
                (*O)->computeLoss();
                int ind = i / n_update;
                storeLosses(j,ind) = (*O)->getLoss();
                std::cout << " " << storeLosses(j,ind);
                ++j;
            }
            std::cout << "\n";
        }
        std::cout << "\tIteration "<< i+1 <<"/"<< n_iter << " completed in "<< (float)omp_get_wtime() - start<< " seconds." << std::endl;
    }
    storeLosses.save(arma::hdf5_name(filename,"dataset",arma::hdf5_opts::trans));
    for (factors::iterator F=Fs.begin(); F!=Fs.end();++F){
        (*F)->writeToFile();
    }
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
