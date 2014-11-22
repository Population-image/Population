#ifndef OPERATORUNIFORMINTDISTRIBUTION_H
#define OPERATORUNIFORMINTDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorUniformIntDistribution : public COperator
{
public:
    OperatorUniformIntDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORUNIFORMINTDISTRIBUTION_H
