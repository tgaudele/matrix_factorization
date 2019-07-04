#ifndef NMTFOBJECTIVE_H
#define NMTFOBJECTIVE_H

#include "sidefactor.h"
#include "centralfactor.h"

class nmtfObjective : public virtual objective
{
public:
    nmtfObjective() {}
    nmtfObjective(double * , const arma::SizeMat &, sideFactor*, centralFactor*, sideFactor*);
    ~nmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);

private:
    arma::mat U, V, W;
};

#endif // NMTFOBJECTIVE_H
