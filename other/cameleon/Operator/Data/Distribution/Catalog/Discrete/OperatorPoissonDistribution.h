#ifndef OPERATORPOISSONDISTRIBUTION_H
#define OPERATORPOISSONDISTRIBUTION_H
#include"COperator.h"
#include<DataDistribution.h>

class OperatorPoissonDistribution : public COperator
{
public:
    OperatorPoissonDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORPOISSONDISTRIBUTION_H
