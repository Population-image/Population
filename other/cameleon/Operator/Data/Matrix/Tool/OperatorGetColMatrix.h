#ifndef OPERATORGETCOLMATRIX_H
#define OPERATORGETCOLMATRIX_H

#include"COperator.h"
class OperatorGetColMatrix: public COperator
{
public:
    OperatorGetColMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORGETCOLMATRIX_H
