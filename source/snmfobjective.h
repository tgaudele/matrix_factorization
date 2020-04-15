#ifndef NMFOBJECTIVE_H
#define NMFOBJECTIVE_H

#include "factor.h"

class nmfObjective : public virtual objective
{
public:
    nmfObjective() {}
    nmfObjective(double *target, const arma::SizeMat &size, factor*, bool = false,double scalar=1.0);
    nmfObjective(double *target, const arma::SizeMat &size, factor*, arma::mat &, arma::mat &, arma::mat &,double scalar=1.0);
    nmfObjective(double *target, const arma::SizeMat &size, factor*, std::string, std::string, std::string,double scalar=1.0);
    ~nmfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);
private:
    arma::mat U, V;
};

#endif // NMFOBJECTIVE_H
