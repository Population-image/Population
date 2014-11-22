#ifndef OPERATORSAVEPOINT_H
#define OPERATORSAVEPOINT_H
#include"COperator.h"
class OperatorSavePoint : public COperator
{
public:
    OperatorSavePoint();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSAVEPOINT_H
