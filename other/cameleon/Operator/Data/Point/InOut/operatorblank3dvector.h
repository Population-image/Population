#ifndef OPERATORBLANK3DVECTOR_H
#define OPERATORBLANK3DVECTOR_H

#include"COperator.h"
class OperatorBlank3DPoint : public COperator
{
public:
    OperatorBlank3DPoint();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORBLANK3DVECTOR_H
