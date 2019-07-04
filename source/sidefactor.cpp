#include "sidefactor.h"

void sideFactor::step(double lr)
{
    initialiseDerivatives();

    int j = 0;
    for (objectives::iterator O=Os.begin(); O!=Os.end();++O) {
        unsigned int p = factorPositions[j];
        (*O)->computeLossDerivatives(p,derivativeNumerator,derivativeDenominator);
        ++j;
    }

    if (!regularizeFlag) {
        weights %= (derivativeNumerator + 1e-9)/(weights * derivativeDenominator + 1e-9);
    }
    else {
        weights %= (derivativeNumerator + 1e-9 + 2.0 * (regularizer * weights))/(weights * derivativeDenominator + 1e-9 + 2.0 * (arma::diagmat(arma::sum(regularizer,1)) * weights));
    }
}
