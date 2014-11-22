#ifndef OPERATORDERIVATEDISTRIBUTION_H
#define OPERATORDERIVATEDISTRIBUTION_H
#include<DataDistribution.h>


class OperatorDerivateDistribution : public COperator
{
public:
    OperatorDerivateDistribution();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORDERIVATEDISTRIBUTION_H
