#ifndef FACTOR_H
#define FACTOR_H

#include "objective.h"
typedef std::vector<objective*> objectives;

class factor
{
public:
    factor() {}
    factor(const std::string &str, unsigned int m, unsigned int n, const std::string& initMode);
    ~factor() {}

    void addGraphRegularizer(double* reg,double lambda=1.0);
    void addL2Regularizer(double lambda=1.0);
    void addObjective(objective* o,unsigned int pos);

    void step();

    void randomInitializer();
    void svdInitializer();
    void initialiseWeights();
    void initialiseDerivatives() { derivativeNumerator.fill(0); derivativeDenominator.fill(0); }

    inline unsigned int getNumRows(){ return numRows; }
    inline unsigned int getNumCols(){ return numCols;}
    inline arma::mat* getWeights() { return  &weights; }
    inline double* getMemPtr() { return  weights.memptr(); }

    void writeToFile();

protected:
    std::string name;
    std::string initializer;
    arma::mat weights;

    unsigned int numRows;
    unsigned int numCols;

    double graphRegularizer = 0;
    double l2Regularizer = 0;
    arma::mat regularizer;

    arma::mat derivativeNumerator;
    arma::mat derivativeDenominator;

    objectives Os;
    std::vector<unsigned int> factorPositions;
    unsigned int numObjectives;

};

#endif // FACTOR_H
