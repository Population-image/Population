#ifndef OPERATORGETPOINT_H
#define OPERATORGETPOINT_H
#include"COperator.h"
class OperatorGetPoint : public COperator
{
public:
    OperatorGetPoint();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORGETPOINT_H
