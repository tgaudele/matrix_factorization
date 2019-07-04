#include "seminmfobjective.h"

seminmfObjective::seminmfObjective(double* target, const arma::SizeMat &size, unconstrainedFactor *u, positiveFactor *v) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
}

void seminmfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V.t(), "fro");
}

void seminmfObjective::computeLossDerivatives(arma::mat &target)
{
    target = X * V * arma::pinv(V.t() * V);
}

void seminmfObjective::computeLossDerivatives(unsigned int factorPosition, arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    derivativeNumerator += (X.t() * U);
    derivativeDenominator += (U.t() * U);
}
