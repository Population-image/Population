#ifndef OPERATORPENCILDISTRIBUTION_H
#define OPERATORPENCILDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorPencilDistribution : public COperator
{
public:
    OperatorPencilDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORPENCILDISTRIBUTION_H
