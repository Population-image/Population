#ifndef OPERATORGENERATE2DROTATIONMATRIX_H
#define OPERATORGENERATE2DROTATIONMATRIX_H

#include"COperator.h"
class OperatorGenerate2DRotationMatrix: public COperator
{
public:
    OperatorGenerate2DRotationMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORGENERATE2DROTATIONMATRIX_H
