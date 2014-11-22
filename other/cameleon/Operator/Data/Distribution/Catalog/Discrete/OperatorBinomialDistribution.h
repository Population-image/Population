#ifndef OPERATORBINOMIALDISTRIBUTION_H
#define OPERATORBINOMIALDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorBinomialDistribution : public COperator
{
public:
    OperatorBinomialDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORBINOMIALDISTRIBUTION_H
