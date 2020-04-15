#include "nmfobjective.h"

nmfObjective::nmfObjective(double* target, const arma::SizeMat &size, factor* u, factor* v, bool compute_svd, double scalar) :
    objective(target, size,compute_svd,std::max(u->getNumCols(),v->getNumCols()), scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
}

nmfObjective::nmfObjective(double* target, const arma::SizeMat &size, factor* u, factor* v, arma::mat &p, arma::mat &q, arma::mat &r, double scalar) :
    objective(target, size, p, q, r, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
}

nmfObjective::nmfObjective(double* target, const arma::SizeMat &size, factor* u, factor* v, std::string pathL, std::string pathS, std::string pathR, double scalar) :
    objective(target, size, pathL, pathR, pathS, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
}

void nmfObjective::computeLoss()
{
    lossScore = scaler * arma::norm(X - U * V.t(), "fro");
}

void nmfObjective::computeLossDerivatives(unsigned int factorPosition, arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        derivativeNumerator += scaler * (X * V);
        derivativeDenominator += scaler * (U * V.t() * V);
        break;}
    case 2: {
        derivativeNumerator += scaler * (X.t() * U);
        derivativeDenominator += scaler * (V * U.t() * U);
        break;}
    }
}
