#include "factor.h"

factor::factor(const std::string &str, unsigned int m, unsigned int n)
{
    name = str;
    numRows = m;
    numCols = n;
    initialiseWeights();
}


void factor::addObjective(objective* o, unsigned int pos)
{
    Os.push_back(o);
    factorPositions.push_back(pos);
    numObjectives = Os.size();
}

void factor::addRegularizer(double *reg, const arma::SizeMat &s)
{
    regularizeFlag = true;
    regularizer = arma::mat(reg,s(0),s(1),false);
}

void factor::writeToFile()
{
    weights.save(arma::hdf5_name(name,"dataset",arma::hdf5_opts::trans));
}


void factor::initialiseWeights()
{
    weights = arma::mat(numRows,numCols,arma::fill::randu) + 1e-9;//arma::clamp(arma::mat(numRows,numCols,arma::fill::randn)*0.25 + 0.5,1e-8,1-1e-8);
}

void factor::reset()
{
    weights.randu(); weights += 1e-9;
}
