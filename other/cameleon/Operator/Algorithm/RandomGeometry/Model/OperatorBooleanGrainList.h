#ifndef OPERATORBOOLEANGRAINLIST_H
#define OPERATORBOOLEANGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorBooleanGermGrain : public COperator
{
public:
    OperatorBooleanGermGrain();
    void exec();
    COperator * clone();
};
#endif // OPERATORBOOLEANGRAINLIST_H
