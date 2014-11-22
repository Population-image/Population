#ifndef OPERATORUNIFORMREALDISTRIBUTION_H
#define OPERATORUNIFORMREALDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorUniformRealDistribution : public COperator
{
public:
    OperatorUniformRealDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORUNIFORMREALDISTRIBUTION_H
