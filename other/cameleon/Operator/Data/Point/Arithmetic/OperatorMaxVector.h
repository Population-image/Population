#ifndef OPERATORMAXVECTOR_H
#define OPERATORMAXVECTOR_H

#include"COperator.h"
class OperatorMaxVector  : public COperator
{
public:
    OperatorMaxVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORMAXVECTOR_H
