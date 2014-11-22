#ifndef OPERATORELLIPSOIDGRAINLIST_H
#define OPERATORELLIPSOIDGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorEllipsoidGermGrain : public COperator
{
public:
    OperatorEllipsoidGermGrain();
    void exec();
    COperator * clone();


};

#endif // OPERATORELLIPSOIDGRAINLIST_H
