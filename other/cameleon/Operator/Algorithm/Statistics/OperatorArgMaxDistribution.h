#ifndef OPERATORARGMAXDISTRIBUTION_H
#define OPERATORARGMAXDISTRIBUTION_H

#include"COperator.h"


class OperatorArgMaxDistribution : public COperator
{
public:
    OperatorArgMaxDistribution();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORARGMAXDISTRIBUTION_H
