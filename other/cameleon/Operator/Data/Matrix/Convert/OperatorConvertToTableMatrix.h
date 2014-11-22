#ifndef OPERATORCONVERTTOTABLEMATRIX_H
#define OPERATORCONVERTTOTABLEMATRIX_H

#include"COperator.h"
class OperatorConvertToTableMatrix: public COperator
{
public:
    OperatorConvertToTableMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORCONVERTTOTABLEMATRIX_H
