#ifndef NMTFOBJECTIVE_H
#define NMTFOBJECTIVE_H

#include "factor.h"
#include "factor.h"

class nmtfObjective : public virtual objective
{
public:
    nmtfObjective() {}
    nmtfObjective(double * , const arma::SizeMat &, factor*, factor*, factor*, bool = false,double scalar=1.0);
    nmtfObjective(double * , const arma::SizeMat &, factor*, factor*, factor*, arma::mat &, arma::mat &, arma::mat &,double scalar=1.0);
    nmtfObjective(double * , const arma::SizeMat &, factor*, factor*, factor*, std::string, std::string, std::string,double scalar=1.0);
    ~nmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);

private:
    arma::mat U, V, W;
};

#endif // NMTFOBJECTIVE_H
