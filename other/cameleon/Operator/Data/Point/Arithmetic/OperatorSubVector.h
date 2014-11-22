#ifndef OPERATORSUBVECTOR_H
#define OPERATORSUBVECTOR_H

#include"COperator.h"
class OperatorSubVector  : public COperator
{
public:
    OperatorSubVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSUBVECTOR_H
