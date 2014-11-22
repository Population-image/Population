#ifndef OPERATORIDENTITYMATRIX_H
#define OPERATORIDENTITYMATRIX_H

#include"COperator.h"
class OperatorIdentityMatrix: public COperator
{
public:
    OperatorIdentityMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORIDENTITYMATRIX_H
