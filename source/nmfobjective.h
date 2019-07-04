#ifndef NMFOBJECTIVE_H
#define NMFOBJECTIVE_H

#include "sidefactor.h"
#include "centralfactor.h"

class nmfObjective : public virtual objective
{
public:
    nmfObjective() {}
    nmfObjective(double *target, const arma::SizeMat &size, sideFactor *u, sideFactor *v);
    ~nmfObjective() {}

    void computeLoss();
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);
private:
    arma::mat U, V;
};

#endif // NMFOBJECTIVE_H
