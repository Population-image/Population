#ifndef OPERATORFOFXDISTRIBUTION_H
#define OPERATORFOFXDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorfofxDistribution : public COperator
{
public:
    OperatorfofxDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORFOFXDISTRIBUTION_H
