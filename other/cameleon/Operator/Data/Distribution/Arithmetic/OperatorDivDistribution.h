#ifndef OPERATORDIVDISTRIBUTION_H
#define OPERATORDIVDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorDivDistribution : public COperator
{
public:
    OperatorDivDistribution();
    void exec();
    COperator * clone();
};


#endif // OPERATORDIVDISTRIBUTION_H
