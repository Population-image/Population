#ifndef OPERATORBOXDISTRIBUTION_H
#define OPERATORBOXDISTRIBUTION_H
#include<COperator.h>
#include<DataDistribution.h>

class OperatorBoxDistribution : public COperator
{
public:
    OperatorBoxDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORBOXDISTRIBUTION_H
