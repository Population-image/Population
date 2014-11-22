#ifndef OPERATORLOADMATRIX_H
#define OPERATORLOADMATRIX_H

#include"COperator.h"
class OperatorLoadMatrix: public COperator
{
public:
    OperatorLoadMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORLOADMATRIX_H
