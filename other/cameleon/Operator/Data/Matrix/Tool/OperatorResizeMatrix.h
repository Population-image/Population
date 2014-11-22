#ifndef OPERATORRESIZEMATRIX_H
#define OPERATORRESIZEMATRIX_H

#include"COperator.h"
class OperatorResizeMatrix: public COperator
{
public:
    OperatorResizeMatrix();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORRESIZEMATRIX_H
