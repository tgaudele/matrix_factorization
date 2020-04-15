#include "objective.h"

void nnsvd(arma::mat &M, arma::mat &N)
{
    for (int i=0; i< arma::size(M)(1);++i) {
        arma::mat pos = arma::max(N.col(i),M.col(i));
        arma::mat neg = arma::max(-N.col(i),M.col(i));

        if (arma::norm(pos) > arma::norm(neg)) {
            M.col(i) = pos;
        }
        else {
            M.col(i) = neg;
        }
    }
}

void objective::getNNSVD(unsigned int ind, arma::mat &G)
{
    int k = G.n_cols;
    arma::mat temp = arma::zeros(arma::size(G));
    arma::mat T;
    switch (ind) {
        case 0:
            T = L(arma::span::all,arma::span(0,k-1)) * arma::diagmat(arma::sqrt(s(arma::span(0,k-1))));
            nnsvd(temp, T);
            G += temp;
            break;
        case 1:
            G += arma::eye(arma::size(G));
            break;
        case 2:
            T = R(arma::span::all,arma::span(0,k-1)) * arma::diagmat(arma::sqrt(s(arma::span(0,k-1))));
            nnsvd(temp,T);
            G += temp;
            break;
    }
}

void completeMatrix(arma::mat &M, int k)
{
    int i = k - (int)M.n_cols;
    if (i > 0) {
        M.insert_cols((int)M.n_cols,i);
    }
}

void completeVector(arma::vec &v, int k)
{
    int i = k - (int)v.n_elem;
    if (i > 0) {
        v.insert_cols((int)v.n_elem,i);
    }
}

void objective::checkSVD(int k)
{
        completeMatrix(L,k);
        completeMatrix(R,k);
        completeVector(s,k);
}
