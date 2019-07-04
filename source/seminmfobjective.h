#ifndef SEMINMFOBJECTIVE_H
#define SEMINMFOBJECTIVE_H

#include "unconstrainedfactor.h"
#include "positivefactor.h"

class seminmfObjective : public virtual objective
{
public:
    seminmfObjective() {}
    seminmfObjective(double *target, const arma::SizeMat &size, unconstrainedFactor *u, positiveFactor *v);
    ~seminmfObjective() {}

    void computeLoss();
    void computeLossDerivatives(arma::mat&);
    void computeLossDerivatives(unsigned int, arma::mat&, arma::mat&);
private:
    arma::mat U, V;
};

#endif // SEMINMFOBJECTIVE_H
