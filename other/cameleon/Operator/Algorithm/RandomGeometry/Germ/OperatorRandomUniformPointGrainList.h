#ifndef OPERATORRANDOMUNIFORMPOINTGRAINLIST_H
#define OPERATORRANDOMUNIFORMPOINTGRAINLIST_H
#include"COperator.h"
class OperatorRandomUniformPointGermGrain : public COperator
{
public:
    OperatorRandomUniformPointGermGrain();
    void exec();
    COperator * clone();

};

#endif // OPERATORRANDOMUNIFORMPOINTGRAINLIST_H
