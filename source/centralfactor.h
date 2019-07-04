#ifndef CENTRALFACTOR_H
#define CENTRALFACTOR_H

#include "factor.h"

class centralFactor : public virtual factor
{
public:
    centralFactor() {}
    centralFactor(const std::string &str,unsigned int m,unsigned int n) : factor(str,m,n) {derivativeNumerator.set_size(numRows,numCols); derivativeDenominator.set_size(numRows,numCols); }
    ~centralFactor() {}

    inline void initialiseDerivatives() { derivativeNumerator.fill(0); derivativeDenominator.fill(0); }
    void step(double lr);
};

#endif // CENTRALFACTOR_H
