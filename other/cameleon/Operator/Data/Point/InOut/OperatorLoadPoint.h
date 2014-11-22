#ifndef OPERATORLOADPOINT_H
#define OPERATORLOADPOINT_H

#include"COperator.h"
class OperatorLoadPoint : public COperator
{
public:
    OperatorLoadPoint();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORLOADPOINT_H
