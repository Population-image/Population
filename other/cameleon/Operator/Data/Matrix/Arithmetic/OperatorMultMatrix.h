#ifndef OPERATORMULTMATRIX_H
#define OPERATORMULTMATRIX_H

#include"COperator.h"
class OperatorMultMatrix: public COperator
{
public:
    OperatorMultMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORMULTMATRIX_H
