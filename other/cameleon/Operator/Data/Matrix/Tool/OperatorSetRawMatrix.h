#ifndef OPERATORSETRAWMATRIX_H
#define OPERATORSETRAWMATRIX_H

#include"COperator.h"
class OperatorSetRawMatrix: public COperator
{
public:
    OperatorSetRawMatrix();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORSETRAWMATRIX_H
