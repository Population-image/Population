#ifndef OPERATORSIZEMATRIX_H
#define OPERATORSIZEMATRIX_H

#include"COperator.h"
class OperatorSizeMatrix: public COperator
{
public:
    OperatorSizeMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORSIZEMATRIX_H
