#ifndef OPERATORRESIZEPOINT_H
#define OPERATORRESIZEPOINT_H

#include"COperator.h"
class OperatorResizePoint : public COperator
{
public:
    OperatorResizePoint();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORRESIZEPOINT_H
