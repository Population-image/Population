#ifndef OPERATORDIRACDISTRIBUTION_H
#define OPERATORDIRACDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorDiracDistribution : public COperator
{
public:
    OperatorDiracDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORDIRACDISTRIBUTION_H
