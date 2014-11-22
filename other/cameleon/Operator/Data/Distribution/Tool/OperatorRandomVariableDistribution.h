#ifndef OPERATORRANDOMVARIABLEDISTRIBUTION_H
#define OPERATORRANDOMVARIABLEDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorRandomVariableDistribution : public COperator
{
public:
    OperatorRandomVariableDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORRANDOMVARIABLEDISTRIBUTION_H
