#ifndef OPERATORHARDCOREFILTERGRAINLIST_H
#define OPERATORHARDCOREFILTERGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorHardCoreGermGrain : public COperator
{
public:
    OperatorHardCoreGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,double radius)
        {

            RandomGeometry::hardCoreFilter(*in,radius);
        }
    };

};

#endif // OPERATORHARDCOREFILTERGRAINLIST_H
