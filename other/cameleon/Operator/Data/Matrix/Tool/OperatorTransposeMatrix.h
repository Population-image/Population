#ifndef OPERATORTRANSPOSEMATRIX_H
#define OPERATORTRANSPOSEMATRIX_H

#include"COperator.h"
class OperatorTransposeMatrixx : public COperator
{
public:
    OperatorTransposeMatrixx();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORTRANSPOSEMATRIX_H
