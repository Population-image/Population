#ifndef RANDOMWALKGRAINLIST_H
#define RANDOMWALKGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorRandomWalkGermGrain : public COperator
{
public:
    OperatorRandomWalkGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,double radius)
        {
            RandomGeometry::randomWalk(*in,radius);
        }
    };

};
#endif // RANDOMWALKGRAINLIST_H
