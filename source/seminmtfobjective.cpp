#include "seminmtfobjective.h"

seminmtfObjective::seminmtfObjective(double* target, const arma::SizeMat &size, positiveFactor *u, unconstrainedFactor *v, positiveFactor *w) :
    objective(target, size),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false)),
    W(arma::mat(w->getMemPtr(),w->getNumRows(),w->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    w->addObjective(this,2);
}

void seminmtfObjective::computeLoss()
{
    lossScore = arma::norm(X - U * V * W.t(), "fro");
}

void seminmtfObjective::computeLossDerivatives(arma::mat &target)
{
    target = arma::pinv( U.t() * U ) * U.t() * X * W * arma::pinv(W.t() * W);
}

void seminmtfObjective::computeLossDerivatives(unsigned int factorPosition,arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        arma::mat temp =  W * V.t();
        derivativeNumerator += (X * temp);
        derivativeDenominator += (temp.t() * temp);
        break;}
    case 2: {
        arma::mat temp = U * V;
        derivativeNumerator += (X.t() * temp);
        derivativeDenominator += (temp.t() * temp);
        break;}
    }
}
