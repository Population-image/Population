#ifndef OPERATORINTEGRALDISTRIBUTION_H
#define OPERATORINTEGRALDISTRIBUTION_H

#include<DataDistribution.h>

class OperatorIntegralDistribution : public COperator
{
public:
    OperatorIntegralDistribution();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORINTEGRALDISTRIBUTION_H
