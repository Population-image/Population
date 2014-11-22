#ifndef OPERATORCONVERTPROBABILITYDISTRIBUTIONDISTRIBUTION_H
#define OPERATORCONVERTPROBABILITYDISTRIBUTIONDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorConvertProbabilityDistributionDistribution : public COperator
{
public:
    OperatorConvertProbabilityDistributionDistribution();
    void exec();
    void initState();
    COperator * clone();
};

#endif // OPERATORCONVERTPROBABILITYDISTRIBUTIONDISTRIBUTION_H
