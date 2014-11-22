#ifndef OPERATORCONTINUOUSTOLATTICEGRAINLIST_H
#define OPERATORCONTINUOUSTOLATTICEGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorContinuousToLatticeGermGrain : public COperator
{
public:
    OperatorContinuousToLatticeGermGrain();
    void exec();
    COperator * clone();
    void initState();

};
#endif // OPERATORCONTINUOUSTOLATTICEGRAINLIST_H
