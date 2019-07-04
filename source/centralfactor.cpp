#include "centralfactor.h"

void centralFactor::step(double lr)
{
    initialiseDerivatives();

    int j = 0;
    for (objectives::iterator O=Os.begin(); O!=Os.end();++O) {
        unsigned int p = factorPositions[j];
        (*O)->computeLossDerivatives(p,derivativeNumerator,derivativeDenominator);
        ++j;
    }

    if (!regularizeFlag) {
        weights %= (derivativeNumerator+1e-9)/(derivativeDenominator+1e-9);
    }
    else {
        weights %= (derivativeNumerator + 1e-9 + 2.0 * (regularizer * weights))/(derivativeDenominator + 1e-9 + 2.0 * (diagmat(arma::sum(regularizer,1)) * weights));
    }
}
