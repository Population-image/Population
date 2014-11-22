#ifndef OPERATORGETRAWMATRIX_H
#define OPERATORGETRAWMATRIX_H


#include"COperator.h"
class OperatorGetRawMatrix: public COperator
{
public:
    OperatorGetRawMatrix();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORGETRAWMATRIX_H
