#ifndef OPERATORMULTVECTORSCALAR_H
#define OPERATORMULTVECTORSCALAR_H

#include"COperator.h"
class OperatorMultVectorScalarVector  : public COperator
{
public:
    OperatorMultVectorScalarVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORMULTVECTORSCALAR_H
