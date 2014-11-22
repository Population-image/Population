#ifndef OPERATORTRACEMATRIX_H
#define OPERATORTRACEMATRIX_H

#include"COperator.h"
class OperatorTraceMatrix: public COperator
{
public:
    OperatorTraceMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORTRACEMATRIX_H
