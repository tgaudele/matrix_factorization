#ifndef SEMISNMTFOBJECTIVE_H
#define SEMISNMTFOBJECTIVE_H

#include "positivefactor.h"
#include "unconstrainedfactor.h"

class semisnmtfObjective : public virtual objective
{
public:
    semisnmtfObjective() {}
    semisnmtfObjective(double *target, const arma::SizeMat &size, positiveFactor*, unconstrainedFactor*);
    ~semisnmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(arma::mat &);
    void computeLossDerivatives(unsigned int, arma::mat &, arma::mat &);
private:
    arma::mat U, V;
};

#endif // SEMISNMTFOBJECTIVE_H
