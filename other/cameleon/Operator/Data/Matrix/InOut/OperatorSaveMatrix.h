#ifndef OPERATORSAVEMATRIX_H
#define OPERATORSAVEMATRIX_H

#include"COperator.h"
class OperatorSaveMatrix: public COperator
{
public:
    OperatorSaveMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSAVEMATRIX_H
