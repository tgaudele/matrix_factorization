#include "snmfobjective.h"

snmfObjective::snmfObjective(double* target, const arma::SizeMat &size, factor* u, bool compute_svd, double scalar) :
    objective(target, size,compute_svd,u->getNumCols(), scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
{
    u->addObjective(this,0);
}

snmfObjective::snmfObjective(double* target, const arma::SizeMat &size, factor* u, arma::mat &p, arma::mat &q, arma::mat &r, double scalar) :
    objective(target, size, p, q, r, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
{
    u->addObjective(this,0);
}

snmfObjective::snmfObjective(double* target, const arma::SizeMat &size, factor* u, std::string pathL, std::string pathS, std::string pathR, double scalar) :
    objective(target, size, pathL, pathR, pathS, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false))
{
    u->addObjective(this,0);
}

void snmfObjective::computeLoss()
{
    lossScore = scaler * arma::norm(X - U * U.t(), "fro");
}

void snmfObjective::computeLossDerivatives(unsigned int factorPosition, arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    derivativeNumerator += scaler * (X * U);
    derivativeDenominator += scaler * (U * U.t() * U);
}
