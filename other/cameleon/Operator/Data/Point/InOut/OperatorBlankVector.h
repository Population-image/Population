#ifndef OPERATORBLANKVECTOR_H
#define OPERATORBLANKVECTOR_H


#include"COperator.h"
class OperatorBlankPoint : public COperator
{
public:
    OperatorBlankPoint();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};


#endif // OPERATORBLANKVECTOR_H
