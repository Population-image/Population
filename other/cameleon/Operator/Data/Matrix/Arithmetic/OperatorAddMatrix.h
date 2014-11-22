#ifndef OPERATORADDMATRIX_H
#define OPERATORADDMATRIX_H
#include"COperator.h"
class OperatorAddMatrix: public COperator
{
public:
    OperatorAddMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORADDMATRIX_H
