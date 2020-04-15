#include "nmtfobjective.h"

nmtfObjective::nmtfObjective(double* target, const arma::SizeMat &size, factor *u, factor *v, factor *w, bool compute_svd, double scalar) :
    objective(target, size, compute_svd, std::max(u->getNumCols(),w->getNumCols()), scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false)),
    W(arma::mat(w->getMemPtr(),w->getNumRows(),w->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    w->addObjective(this,2);
}

nmtfObjective::nmtfObjective(double* target, const arma::SizeMat &size, factor *u, factor *v, factor *w, arma::mat &p, arma::mat &q, arma::mat &r, double scalar) :
    objective(target, size, p, q, r, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    w->addObjective(this,2);
}

nmtfObjective::nmtfObjective(double* target, const arma::SizeMat &size, factor *u, factor *v, factor *w, std::string pathL, std::string pathS, std::string pathR, double scalar) :
    objective(target, size, pathL, pathR, pathS, scalar),
    U(arma::mat(u->getMemPtr(),u->getNumRows(),u->getNumCols(),false)),
    V(arma::mat(v->getMemPtr(),v->getNumRows(),v->getNumCols(),false))
{
    u->addObjective(this,0);
    v->addObjective(this,1);
    w->addObjective(this,2);
}

void nmtfObjective::computeLoss()
{
    lossScore = scaler * arma::norm(X - U * V * W.t(), "fro");
}

void nmtfObjective::computeLossDerivatives(unsigned int factorPosition,arma::mat &derivativeNumerator, arma::mat &derivativeDenominator)
{
    switch (factorPosition) {
    case 0: {
        arma::mat temp =  W * V.t();
        derivativeNumerator += scaler * (X * temp);
        derivativeDenominator += scaler * (U * temp.t() * temp);
        break;}
    case 1: {
        arma::mat temp = U.t();
        derivativeNumerator += scaler * (temp * X * W);
        derivativeDenominator += scaler * (temp * U * V * W.t() * W);
        break;}
    case 2: {
        arma::mat temp = U * V;
        derivativeNumerator += scaler * (X.t() * temp);
        derivativeDenominator += scaler * (V * temp.t() * temp);
        break;}
    }
}
