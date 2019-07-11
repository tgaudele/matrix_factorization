#include "objective.h"

void objective::getSVDNMF(unsigned int ind, arma::mat &G)
{
    int k = arma::size(G)(1);
    arma::mat temp;
    switch (ind) {
        case 0:
            temp = arma::abs(L(arma::span::all,arma::span(0,k-1)));
            G += temp * arma::diagmat(arma::sqrt(s(arma::span(0,k-1))));
            break;
        case 1:
            G += arma::eye(arma::size(G));
            break;
        case 2:
            temp = arma::abs(R(arma::span::all,arma::span(0,k-1)));
            G += temp * arma::diagmat(arma::sqrt(s(arma::span(0,k-1))));
            break;
    }
}

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
    int k = arma::size(G)(1);
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
