#ifndef SNMTFOBJECTIVE_H
#define SNMTFOBJECTIVE_H

#include "factor.h"

class snmtfObjective : public virtual objective
{
public:
    snmtfObjective() {}
    snmtfObjective(double *target, const arma::SizeMat &size, factor*, factor*, bool = false,double scalar=1.0);
    snmtfObjective(double *target, const arma::SizeMat &size, factor*, factor*, arma::mat &, arma::mat &, arma::mat &,double scalar=1.0);
    snmtfObjective(double *target, const arma::SizeMat &size, factor*, factor*, std::string, std::string, std::string,double scalar=1.0);
    ~snmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat &, arma::mat &);
private:
    arma::mat U, V;
};

#endif // SNMTFOBJECTIVE_H
