#ifndef OPERATORGETMATRIX_H
#define OPERATORGETMATRIX_H


#include"COperator.h"
class OperatorGetMatrix: public COperator
{
public:
    OperatorGetMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORGETMATRIX_H
