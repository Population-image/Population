#ifndef OPERATORADDVECTOR_H
#define OPERATORADDVECTOR_H

#include"COperator.h"
class OperatorAddVector  : public COperator
{
public:
    OperatorAddVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORADDVECTOR_H
