#include "unconstrainedfactor.h"

void unconstrainedFactor::step(double lr)
{
    for (objectives::iterator O=Os.begin(); O!=Os.end();++O) {
        (*O)->computeLossDerivatives(weights);
    }
}
