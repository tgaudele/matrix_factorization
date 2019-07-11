#ifndef NMTF_INTEGRATION_H
#define NMTF_INTEGRATION_H

#include "factor.h"
#include "nmfobjective.h"
#include "nmtfobjective.h"
#include "snmtfobjective.h"
#include "seminmfobjective.h"
#include "seminmtfobjective.h"
#include "semisnmtfobjective.h"

class integration
{
public:
    integration() {}
    integration(objectives &o_list, factors &f_list): Os(o_list), Fs(f_list), numObjectives(o_list.size()), numFactors(f_list.size()) {}
    ~integration() {}

    void optimize(const unsigned int n_iter, const unsigned int n_update, const std::string &filename);
    double computeLoss();

private:
    objectives Os;
    factors Fs;
    unsigned int numObjectives;
    unsigned int numFactors;
};

#endif // NMTF_INTEGRATION_H
