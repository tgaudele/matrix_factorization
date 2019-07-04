#ifndef SEMINMTFOBJECTIVE_H
#define SEMINMTFOBJECTIVE_H

#include "unconstrainedfactor.h"
#include "positivefactor.h"

class seminmtfObjective : public virtual objective
{
public:
    seminmtfObjective() {}
    seminmtfObjective(double * , const arma::SizeMat &, positiveFactor*, unconstrainedFactor*, positiveFactor*);
    ~seminmtfObjective() {}

    void computeLoss();
    void computeLossDerivatives(arma::mat&);
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);

private:
    arma::mat U, V, W;
};

#endif // SEMINMTFOBJECTIVE_H
