#ifndef OPERATOREIGENVECTORMATRIX_H
#define OPERATOREIGENVECTORMATRIX_H

#include"COperator.h"
class OperatorEigenVectorMatrix: public COperator
{
public:
    OperatorEigenVectorMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATOREIGENVECTORMATRIX_H
