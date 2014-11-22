#ifndef OPERATORDEADLEAVEBLACKANDWHITEGRAINLIST_H
#define OPERATORDEADLEAVEBLACKANDWHITEGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorDeadLeaveGermGrain : public COperator
{
public:
    OperatorDeadLeaveGermGrain();
    void exec();
    COperator * clone();
};

#endif // OPERATORDEADLEAVEBLACKANDWHITEGRAINLIST_H
