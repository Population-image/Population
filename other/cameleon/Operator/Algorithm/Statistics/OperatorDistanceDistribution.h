#ifndef OPERATORDISTANCEDISTRIBUTION_H
#define OPERATORDISTANCEDISTRIBUTION_H

#include<DataDistribution.h>

class OperatorDistanceDistribution : public COperator
{
public:
    OperatorDistanceDistribution();
    void exec();
    COperator * clone();
    void initState();
};

#endif // OPERATORDISTANCEDISTRIBUTION_H
