#ifndef OPERATORSIZEPOINT_H
#define OPERATORSIZEPOINT_H

#include"COperator.h"
class OperatorSizePoint : public COperator
{
public:
    OperatorSizePoint();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORSIZEPOINT_H
