#ifndef OPERATORCONVERTFROMTABLEMATRIX_H
#define OPERATORCONVERTFROMTABLEMATRIX_H

#include"COperator.h"
class OperatorConvertFromTableMatrix: public COperator
{
public:
    OperatorConvertFromTableMatrix();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORCONVERTFROMTABLEMATRIX_H
