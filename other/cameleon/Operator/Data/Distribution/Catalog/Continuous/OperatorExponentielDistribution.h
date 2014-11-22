#ifndef OPERATOREXPONENTIELDISTRIBUTION_H
#define OPERATOREXPONENTIELDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorExponentielDistribution : public COperator
{
public:
    OperatorExponentielDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATOREXPONENTIELDISTRIBUTION_H
