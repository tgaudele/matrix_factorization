#include "factor.h"

factor::factor(const std::string &str, unsigned int m, unsigned int n, const std::string &initMode)
{
    name = str;
    numRows = m;
    numCols = n;
    weights = arma::mat(m,n,arma::fill::zeros);
    initializer = initMode;
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

void factor::randomInitializer()
{
    weights.randu(); weights += 1e-9;
}

void factor::svdnmfInitializer()
{
    weights.fill(0);
    for (unsigned int i; i < Os.size(); ++i ) {
        Os[i]->getSVDNMF(factorPositions[i],weights);
    }
    weights /= Os.size(); weights += 1e-9;
}

void factor::nnsvdInitializer()
{
    weights.fill(0);
    for (unsigned int i; i < Os.size(); ++i ) {
        Os[i]->getNNSVD(factorPositions[i],weights);
    }
    weights /= Os.size(); weights += 1e-9;
}

void factor::initialiseWeights()
{
    if (initializer.compare("nnsvd") == 0) {
        nnsvdInitializer();
    }
    else if (initializer.compare("svd-nmf") == 0) {
        svdnmfInitializer();
    }
    else if (initializer.compare("rand") == 0) {
        randomInitializer();
    }
    else {
        std::cerr << "Initializer unsupported, choose one of {svdnmf,nnsvd,rand}";
    }
}
