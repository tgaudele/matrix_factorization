#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>

class factor;
typedef std::vector<factor*> factors;

class objective
{
public:
    objective() {}
    objective(double* target, const arma::SizeMat &targetSize) : X(arma::mat(target,targetSize(0),targetSize(1),false)) {}
    ~objective() {}

   void virtual computeLossDerivatives(unsigned int, arma::mat&, arma::mat&) {}
   void virtual computeLossDerivatives(arma::mat&) {}
   void virtual computeLoss() {}
   inline double getLoss() { return lossScore; }


protected:
    factors Fs;
    arma::mat X;
    double lossScore;
};

#endif // OBJECTIVE_H
