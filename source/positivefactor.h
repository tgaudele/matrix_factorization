#ifndef POSITIVEFACTOR_H
#define POSITIVEFACTOR_H

#include "factor.h"

class positiveFactor : public virtual factor
{
public:
    positiveFactor() {}
    positiveFactor(const std::string &str,unsigned int m, unsigned int n, const std::string &init) : factor(str,m,n,init) { derivativeNumerator.set_size(numRows,numCols); derivativeDenominator.set_size(numCols,numCols); }
    ~positiveFactor() {}

    void step(double lr);
    inline void initialiseDerivatives() { derivativeNumerator.fill(0); derivativeDenominator.fill(0); }
};

#endif // POSITIVEFACTOR_H
