#ifndef OPERATORSAVEDISTRIBUTION_H
#define OPERATORSAVEDISTRIBUTION_H

#include"COperator.h"

class OperatorSaveDistribution: public COperator
{
public:
    OperatorSaveDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORSAVEDISTRIBUTION_H
