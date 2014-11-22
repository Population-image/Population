#ifndef OPERATORARGMINDISTRIBUTION_H
#define OPERATORARGMINDISTRIBUTION_H

#include"COperator.h"


class OperatorArgMinDistribution : public COperator
{
public:
    OperatorArgMinDistribution();
    void exec();
    COperator * clone();
        void initState();
};


#endif // OPERATORARGMINDISTRIBUTION_H
