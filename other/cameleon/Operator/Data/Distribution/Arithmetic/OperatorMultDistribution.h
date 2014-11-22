#ifndef OPERATORMULTDISTRIBUTION_H
#define OPERATORMULTDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorMultDistribution : public COperator
{
public:
    OperatorMultDistribution();
    void exec();
    COperator * clone();
};


#endif // OPERATORMULTDISTRIBUTION_H
