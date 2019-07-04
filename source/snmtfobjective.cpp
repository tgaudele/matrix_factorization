#include "snmtfobjective.h"

snmtfObjective::snmtfObjective(double* target, const arma::SizeMat &size, sideFactor* u, centralFactor* v) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
}

void snmtfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V * U.t(), "fro");
}

void snmtfObjective::computeLossDerivatives(unsigned int factorPosition,arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        derivativeNumerator += (X * U * (V + V.t()));
        derivativeDenominator += (V * U.t() * U * (V + V.t()));
        break;}
    case 1: {
        arma::mat temp = U.t() * U;
        derivativeNumerator = (U.t() * X * U);
        derivativeDenominator += (temp * V * temp);
        break;}
    }

}
