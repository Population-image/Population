#ifndef OPERATORTRANSPARENCYGRAINLIST_H
#define OPERATORTRANSPARENCYGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorTransparencyGermGrain : public COperator
{
public:
    OperatorTransparencyGermGrain();
    void exec();
    COperator * clone();
};
#endif // OPERATORTRANSPARENCYGRAINLIST_H
