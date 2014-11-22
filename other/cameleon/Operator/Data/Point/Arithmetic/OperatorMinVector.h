#ifndef OPERATORMINVECTOR_H
#define OPERATORMINVECTOR_H

#include"COperator.h"
class OperatorMinVector  : public COperator
{
public:
    OperatorMinVector();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORMINVECTOR_H
