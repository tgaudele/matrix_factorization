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
    objective(double* target, const arma::SizeMat &targetSize,bool compute_svd,int k, double scalar=1.0) : X(arma::mat(target,targetSize(0),targetSize(1),false))
    {
        scaler = scalar;
        if (compute_svd) {
            arma::svds(L,s,R,arma::sp_mat(X),k);
            checkSVD(k);
        }
    }

    objective(double* target, const arma::SizeMat &targetSize, arma::mat &p, arma::mat &q, arma::mat &r, double scalar=1.0) : X(arma::mat(target,targetSize(0),targetSize(1),false))
    {
        scaler = scalar;
        L = p; s = q; R = r;
        checkSVD(std::max((int)L.n_cols,(int)R.n_cols));
    }

    objective(double* target, const arma::SizeMat &targetSize, std::string pathL, std::string pathS, std::string pathR, double scalar=1.0) : X(arma::mat(target,targetSize(0),targetSize(1),false))
    {
        scaler = scalar;
        L.load(arma::hdf5_name(pathL,"dataset",arma::hdf5_opts::trans));
        R.load(arma::hdf5_name(pathR,"dataset",arma::hdf5_opts::trans));
        s.load(arma::hdf5_name(pathS,"dataset",arma::hdf5_opts::trans));
        checkSVD(std::max((int)L.n_cols,(int)R.n_cols));
    }

    ~objective() {}

   void virtual computeLossDerivatives(unsigned int, arma::mat&, arma::mat&) {}
   void virtual computeLossDerivatives(arma::mat&) {}
   void virtual computeLoss() {}
   inline double getLoss() { return lossScore; }

   void getSVDNMF(unsigned int, arma::mat&);
   void getNNSVD(unsigned int, arma::mat&);
   void checkSVD(int);

protected:
    factors Fs;
    arma::mat X;
    double lossScore;
    arma::mat L;
    arma::vec s;
    arma::mat R;
    double scaler = 1.0;
};

#endif // OBJECTIVE_H
