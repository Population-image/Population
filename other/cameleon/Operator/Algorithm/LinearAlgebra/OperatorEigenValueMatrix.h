#ifndef OPERATOREIGENVALUEMATRIX_H
#define OPERATOREIGENVALUEMATRIX_H

#include"COperator.h"
class OperatorEigenValueMatrix: public COperator
{
public:
    OperatorEigenValueMatrix();
    virtual void exec();
    void initState();
    virtual COperator * clone();
};

#endif // OPERATOREIGENVALUEMATRIX_H
