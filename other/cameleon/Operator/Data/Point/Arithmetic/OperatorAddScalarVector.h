#ifndef OPERATORADDSCALARVECTOR_H
#define OPERATORADDSCALARVECTOR_H

#include"COperator.h"
class OperatorAddScalarVector  : public COperator
{
public:
    OperatorAddScalarVector();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORADDSCALARVECTOR_H
