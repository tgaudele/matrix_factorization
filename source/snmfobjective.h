#ifndef SNMFOBJECTIVE_H
#define SNMFOBJECTIVE_H

#include "factor.h"

class snmfObjective : public virtual objective
{
public:
    snmfObjective() {}
    snmfObjective(double *target, const arma::SizeMat &size, factor*, bool = false,double scalar=1.0);
    snmfObjective(double *target, const arma::SizeMat &size, factor*, arma::mat &, arma::mat &, arma::mat &,double scalar=1.0);
    snmfObjective(double *target, const arma::SizeMat &size, factor*, std::string, std::string, std::string,double scalar=1.0);
    ~snmfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);
private:
    arma::mat U;
};

#endif // SNMFOBJECTIVE_H
