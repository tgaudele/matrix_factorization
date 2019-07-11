#include "nmtfobjective.h"

nmtfObjective::nmtfObjective(double* target, const arma::SizeMat &size, sideFactor *u, centralFactor *v, sideFactor *w) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false)),
    W(arma::mat(w->getMemPtr(),w->getNumRows(),w->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    w->addObjective(this,2);
    int k = std::max(u->getNumCols(),w->getNumCols());
    arma::svds(L,s,R,arma::sp_mat(X),k);
}

void nmtfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V * W.t(), "fro");
}

void nmtfObjective::computeLossDerivatives(unsigned int factorPosition,arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        arma::mat temp =  W * V.t();
        derivativeNumerator += (X * temp);
        derivativeDenominator += (temp.t() * temp);
        break;}
    case 1: {
        arma::mat temp = U.t();
        derivativeNumerator += (temp * X * W);
        derivativeDenominator += (temp * U * V * W.t() * W);
        break;}
    case 2: {
        arma::mat temp = U * V;
        derivativeNumerator += (X.t() * temp);
        derivativeDenominator += (temp.t() * temp);
        break;}
    }
}
