#include "positivefactor.h"

void positiveFactor::step(double lr)
{
    initialiseDerivatives();

    int j = 0;
    for (objectives::iterator O=Os.begin(); O!=Os.end();++O) {
        unsigned int p = factorPositions[j];
        (*O)->computeLossDerivatives(p,derivativeNumerator,derivativeDenominator);
        ++j;
    }

    arma::mat numerator = (weights * (arma::abs(derivativeDenominator) - derivativeDenominator) + arma::abs(derivativeNumerator) + derivativeNumerator)/2.0;
    arma::mat denominator = (weights * (arma::abs(derivativeDenominator) + derivativeDenominator) + arma::abs(derivativeNumerator) - derivativeNumerator)/2.0;

    if (!regularizeFlag) {
        weights %= arma::sqrt((numerator + 1e-9)/(denominator + 1e-9));
    }
    else {
        weights %= arma::sqrt((numerator + 1e-9 + 2.0 * (regularizer * weights))/(denominator + 1e-9 + 2.0 * (arma::diagmat(arma::sum(regularizer,1)) * weights)));
    }
}
