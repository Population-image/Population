#ifndef OPERATORDETERMINANTMATRIX_H
#define OPERATORDETERMINANTMATRIX_H

#include"COperator.h"
class OperatorDeterminantMatrix: public COperator
{
public:
    OperatorDeterminantMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORDETERMINANTMATRIX_H
