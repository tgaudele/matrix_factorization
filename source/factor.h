#ifndef FACTOR_H
#define FACTOR_H

#include "objective.h"
typedef std::vector<objective*> objectives;

class factor
{
public:
    factor() {}
    factor(const std::string &str,unsigned int m, unsigned int n);
    ~factor() {}

    void addRegularizer(double* reg, const arma::SizeMat &s);
    void addObjective(objective* o,unsigned int pos);

    void virtual step(double lr) {}

    void initialiseWeights();
    void virtual initialiseDerivatives() {}
    void reset();

    inline unsigned int getNumRows(){ return numRows; }
    inline unsigned int getNumCols(){ return numCols;}
    inline arma::mat* getWeights() { return  &weights; }
    inline double* getMemPtr() { return  weights.memptr(); }

    void writeToFile();

protected:
    std::string name;
    arma::mat weights;
    unsigned int numRows;
    unsigned int numCols;
    bool regularizeFlag = false;
    arma::mat regularizer;

    arma::mat derivativeNumerator;
    arma::mat derivativeDenominator;

    objectives Os;
    std::vector<unsigned int> factorPositions;
    unsigned int numObjectives;

};

#endif // FACTOR_H
