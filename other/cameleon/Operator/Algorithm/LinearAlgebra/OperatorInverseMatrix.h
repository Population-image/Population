#ifndef OPERATORINVERSEMATRIX_H
#define OPERATORINVERSEMATRIX_H

#include"COperator.h"
class OperatorInverseMatrix: public COperator
{
public:
    OperatorInverseMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORINVERSEMATRIX_H
