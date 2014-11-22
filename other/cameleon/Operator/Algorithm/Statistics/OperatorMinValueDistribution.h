#ifndef OPERATORMINVALUEDISTRIBUTION_H
#define OPERATORMINVALUEDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorMinValueDistribution : public COperator
{
public:
    OperatorMinValueDistribution();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORMINVALUEDISTRIBUTION_H
