#ifndef OPERATORINVERSEDISTRIBUTION_H
#define OPERATORINVERSEDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorInverseDistribution : public COperator
{
public:
    OperatorInverseDistribution();
    void exec();
    COperator * clone();
};


#endif // OPERATORINVERSEDISTRIBUTION_H
