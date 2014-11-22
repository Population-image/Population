#ifndef OPERATORSUBMATRIX_H
#define OPERATORSUBMATRIX_H

#include"COperator.h"
class OperatorSubMatrix: public COperator
{
public:
    OperatorSubMatrix();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORSUBMATRIX_H
