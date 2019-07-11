#ifndef SIDEFACTOR_H
#define SIDEFACTOR_H

#include "factor.h"

class sideFactor : public virtual factor
{
public:
    sideFactor() {}
    sideFactor(const std::string &str,unsigned int m, unsigned int n, const std::string &init) : factor(str,m,n,init) { derivativeNumerator.set_size(numRows,numCols); derivativeDenominator.set_size(numCols,numCols); }
    ~sideFactor() {}

    void step(double lr);
    inline void initialiseDerivatives() { derivativeNumerator.fill(0); derivativeDenominator.fill(0); }
};

#endif // SIDEFACTOR_H
