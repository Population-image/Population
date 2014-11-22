#ifndef COMPUTEDSTATICTICSDISTRIBUTION_H
#define COMPUTEDSTATICTICSDISTRIBUTION_H

#include"COperator.h"

class OperatorComputedStaticticsDistribution: public COperator
{
public:
    OperatorComputedStaticticsDistribution();
    void exec();
    void initState();
    COperator * clone();
};
#endif // COMPUTEDSTATICTICSDISTRIBUTION_H
