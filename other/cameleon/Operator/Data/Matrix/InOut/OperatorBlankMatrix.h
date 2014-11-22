#ifndef OPERATORBLANKMATRIX_H
#define OPERATORBLANKMATRIX_H

#include"COperator.h"
class OperatorBlankMatrix: public COperator
{
public:
    OperatorBlankMatrix();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};

#endif // OPERATORBLANKMATRIX_H
