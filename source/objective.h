#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>

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

   void getSVDNMF(unsigned int, arma::mat&);
   void getNNSVD(unsigned int, arma::mat&);

protected:
    factors Fs;
    arma::mat X;
    double lossScore;
    arma::mat L;
    arma::vec s;
    arma::mat R;
};

#endif // OBJECTIVE_H
