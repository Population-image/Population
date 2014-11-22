#ifndef OPERATORMULTMATRIXSCALAR_H
#define OPERATORMULTMATRIXSCALAR_H

#include"COperator.h"
class OperatorMultMatrixScalarMatrix: public COperator
{
public:
    OperatorMultMatrixScalarMatrix();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORMULTMATRIXSCALAR_H
