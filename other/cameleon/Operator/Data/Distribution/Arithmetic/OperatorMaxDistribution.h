#ifndef OPERATORMAXDISTRIBUTION_H
#define OPERATORMAXDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorMaxDistribution : public COperator
{
public:
    OperatorMaxDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORMAXDISTRIBUTION_H
