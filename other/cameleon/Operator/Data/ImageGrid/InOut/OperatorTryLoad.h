#ifndef OPERATORTRYLOAD_H
#define OPERATORTRYLOAD_H
#include"COperator.h"

class OperatorTryLoad: public COperator
{
public:
    OperatorTryLoad();
    void exec();
    COperator * clone();
};

#endif // OPERATORTRYLOAD_H
