#ifndef OPERATORBLANK2DVECTOR_H
#define OPERATORBLANK2DVECTOR_H

#include"COperator.h"
class OperatorBlank2DPoint : public COperator
{
public:
    OperatorBlank2DPoint();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORBLANK2DVECTOR_H
