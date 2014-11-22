#ifndef OPERATORLOADDISTRIBUTION_H
#define OPERATORLOADDISTRIBUTION_H

#include"COperator.h"

class OperatorLoadDistribution: public COperator
{
public:
    OperatorLoadDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORLOADDISTRIBUTION_H
