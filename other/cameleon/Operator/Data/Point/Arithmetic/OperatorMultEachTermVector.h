#ifndef OPERATORMULTEACHTERMVECTOR_H
#define OPERATORMULTEACHTERMVECTOR_H

#include"COperator.h"
class OperatorMultEachTermVector  : public COperator
{
public:
    OperatorMultEachTermVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORMULTEACHTERMVECTOR_H
