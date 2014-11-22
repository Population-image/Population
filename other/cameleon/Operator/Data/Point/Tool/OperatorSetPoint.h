#ifndef OPERATORSETPOINT_H
#define OPERATORSETPOINT_H

#include"COperator.h"
class OperatorSetPoint : public COperator
{
public:
    OperatorSetPoint();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETPOINT_H
