#ifndef OPERATORMINOVERLAPGRAINLIST_H
#define OPERATORMINOVERLAPGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorMinOverlapGermGrain : public COperator
{
public:
    OperatorMinOverlapGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,double radius)
        {
            RandomGeometry::minOverlapFilter(*in,radius);
        }
    };

};
#endif // OPERATORMINOVERLAPGRAINLIST_H
