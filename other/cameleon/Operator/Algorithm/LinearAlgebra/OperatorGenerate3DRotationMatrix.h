#ifndef OPERATORGENERATE3DROTATIONMATRIX_H
#define OPERATORGENERATE3DROTATIONMATRIX_H

#include"COperator.h"
class OperatorGenerate3DRotationMatrix: public COperator
{
public:
    OperatorGenerate3DRotationMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORGENERATE3DROTATIONMATRIX_H
