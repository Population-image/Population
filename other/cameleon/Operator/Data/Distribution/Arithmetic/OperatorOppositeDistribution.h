#ifndef OPERATOROPPOSITEDISTRIBUTION_H
#define OPERATOROPPOSITEDISTRIBUTION_H
#include"COperator.h"
#include<DataDistribution.h>

class OperatorOppositeDistribution : public COperator
{
public:
    OperatorOppositeDistribution();
    void exec();
    COperator * clone();
};


#endif // OPERATOROPPOSITEDISTRIBUTION_H
