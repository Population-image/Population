#ifndef OPERATORSETCOLMATRIX_H
#define OPERATORSETCOLMATRIX_H

#include"COperator.h"
class OperatorSetColMatrix: public COperator
{
public:
    OperatorSetColMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETCOLMATRIX_H
