#ifndef OPERATORCOMPODISTRIBUTION_H
#define OPERATORCOMPODISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorCompositionDistribution : public COperator
{
public:
    OperatorCompositionDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORCOMPODISTRIBUTION_H
