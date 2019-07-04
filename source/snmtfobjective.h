#ifndef SNMTFOBJECTIVE_H
#define SNMTFOBJECTIVE_H

#include "sidefactor.h"
#include "centralfactor.h"

class snmtfObjective : public virtual objective
{
public:
    snmtfObjective() {}
    snmtfObjective(double *target, const arma::SizeMat &size, sideFactor*, centralFactor*);
    ~snmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat &, arma::mat &);
private:
    arma::mat U, V;
};

#endif // SNMTFOBJECTIVE_H
