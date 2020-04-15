#include "factor.h"

factor::factor(const std::string &str, unsigned int m, unsigned int n, const std::string &initMode)
{
    name = str;
    numRows = m;
    numCols = n;
    weights = arma::mat(m,n,arma::fill::zeros);
    initializer = initMode;
    derivativeNumerator.set_size(numRows,numCols);
    derivativeDenominator.set_size(numRows,numCols);
}


void factor::addObjective(objective* o, unsigned int pos)
{
    Os.push_back(o);
    factorPositions.push_back(pos);
    numObjectives = Os.size();
}

void factor::addGraphRegularizer(double *reg,double lambda)
{
    graphRegularizer = lambda;
    regularizer = arma::mat(reg,numRows,numRows,false);
}

void factor::addL2Regularizer(double lambda)
{
    l2Regularizer = lambda;
}

void factor::writeToFile()
{
    weights.save(arma::hdf5_name(name,"dataset",arma::hdf5_opts::trans));
}

void factor::randomInitializer()
{
    weights.randu(); weights += 1e-9;
}

void factor::svdInitializer()
{
    weights.fill(0);
    for (unsigned int i; i < Os.size(); ++i ) {
        Os[i]->getNNSVD(factorPositions[i],weights);
    }
    weights /= Os.size(); weights += 1e-9;
}

void factor::initialiseWeights()
{
    if (initializer.compare("tsvd") == 0) {
        svdInitializer();
    }
    else if (initializer.compare("random") == 0) {
        randomInitializer();
    }
    else {
        std::cerr << "Initializer unsupported, choose one of {tsvd,random}";
    }
}

void factor::step()
{
    initialiseDerivatives();

    int j = 0;
    for (objectives::iterator O=Os.begin(); O!=Os.end();++O) {
        unsigned int p = factorPositions[j];
        (*O)->computeLossDerivatives(p,derivativeNumerator,derivativeDenominator);
        ++j;
    }

    if (l2Regularizer > 0) { derivativeDenominator += (l2Regularizer * weights);}

    if (graphRegularizer > 0) {
        derivativeNumerator += (graphRegularizer *  regularizer * weights);
        derivativeDenominator += (graphRegularizer *  arma::diagmat(arma::sum(regularizer,1)) * weights);
    }

    weights %= (derivativeNumerator + 1e-9)/(derivativeDenominator + 1e-9);

}
