#ifndef OPERATORADDDISTRIBUTION_H
#define OPERATORADDDISTRIBUTION_H
#include"COperator.h"
#include<DataDistribution.h>

class OperatorAddDistribution : public COperator
{
public:
    OperatorAddDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORADDDISTRIBUTION_H
