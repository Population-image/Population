#ifndef OPERATORSETMATRIX_H
#define OPERATORSETMATRIX_H

#include"COperator.h"
class OperatorSetMatrix: public COperator
{
public:
    OperatorSetMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORSETMATRIX_H
