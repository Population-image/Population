#ifndef OPERATORMULTVECTORMATRIX_H
#define OPERATORMULTVECTORMATRIX_H

#include"COperator.h"
class OperatorMultMatrixVectorMatrix: public COperator
{
public:
    OperatorMultMatrixVectorMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORMULTVECTORMATRIX_H
