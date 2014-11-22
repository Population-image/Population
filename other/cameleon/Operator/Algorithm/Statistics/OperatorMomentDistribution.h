#ifndef OPERATORMOMENTDISTRIBUTION_H
#define OPERATORMOMENTDISTRIBUTION_H

#include<DataDistribution.h>

class OperatorMomentDistribution : public COperator
{
public:
    OperatorMomentDistribution();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORMOMENTDISTRIBUTION_H
