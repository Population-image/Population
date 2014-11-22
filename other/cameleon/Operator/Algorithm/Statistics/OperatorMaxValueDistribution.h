#ifndef OPERATORMAXVALUEDISTRIBUTION_H
#define OPERATORMAXVALUEDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorMaxValueDistribution : public COperator
{
public:
    OperatorMaxValueDistribution();
    void exec();
    COperator * clone();
        void initState();
};

#endif // OPERATORMAXVALUEDISTRIBUTION_H
