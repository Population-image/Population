#ifndef OPERATORORTHOGONALMATRIX_H
#define OPERATORORTHOGONALMATRIX_H

#include"COperator.h"
class OperatorOrthogonalMatrix: public COperator
{
public:
    OperatorOrthogonalMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORORTHOGONALMATRIX_H
