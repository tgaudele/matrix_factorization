#ifndef UNCONSTRAINEDFACTOR_H
#define UNCONSTRAINEDFACTOR_H

#include "factor.h"

class unconstrainedFactor : public virtual factor
{
public:
    unconstrainedFactor() {}
    unconstrainedFactor(const std::string &str,unsigned int m,unsigned int n) : factor(str,m,n) {}
    ~unconstrainedFactor() {}

    inline void initialiseDerivatives() { derivativeNumerator.fill(0); derivativeDenominator.fill(0); }
    void step(double lr);
};

#endif // UNCONSTRAINEDFACTOR_H
