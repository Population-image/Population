#ifndef OPERATORNORMALDISTRIBUTION_H
#define OPERATORNORMALDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorNormalDistribution : public COperator
{
public:
    OperatorNormalDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORNORMALDISTRIBUTION_H
