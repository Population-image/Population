#ifndef OPERATORBOXGRAINLIST_H
#define OPERATORBOXGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorBoxGermGrain : public COperator
{
public:
    OperatorBoxGermGrain();
    void exec();
    COperator * clone();

};
#endif // OPERATORBOXGRAINLIST_H
