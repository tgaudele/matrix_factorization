#include "nmfobjective.h"

nmfObjective::nmfObjective(double* target, const arma::SizeMat &size, sideFactor *u, sideFactor *v) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,2);
    int k = std::max(u->getNumCols(),v->getNumCols());
    arma::svds(L,s,R,arma::sp_mat(X),k);
}

void nmfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V.t(), "fro");
}

void nmfObjective::computeLossDerivatives(unsigned int factorPosition, arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        derivativeNumerator += (X * V);
        derivativeDenominator += (V.t() * V);
        break;}
    case 2: {
        derivativeNumerator += (X.t() * U);
        derivativeDenominator += (U.t() * U);
        break;}
    }
}
