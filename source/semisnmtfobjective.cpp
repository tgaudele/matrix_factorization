#include "semisnmtfobjective.h"

semisnmtfObjective::semisnmtfObjective(double* target, const arma::SizeMat &size, positiveFactor *u, unconstrainedFactor *v) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    int k = u->getNumCols();
    arma::svds(L,s,R,arma::sp_mat(X),k);
}

void semisnmtfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V * U.t(), "fro");
}

void semisnmtfObjective::computeLossDerivatives(arma::mat &target)
{
    arma::mat temp = arma::pinv(U.t() * U);
    target = temp * U.t() * X * U * temp;
}

void semisnmtfObjective::computeLossDerivatives(unsigned int factorPosition,arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
        derivativeNumerator += (X * U * (V + V.t()));
        derivativeDenominator += (V * U.t() * U * (V + V.t()));
}
