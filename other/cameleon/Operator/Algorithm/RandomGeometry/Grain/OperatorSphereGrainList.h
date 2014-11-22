#ifndef OPERATORSPHEREGRAINLIST_H
#define OPERATORSPHEREGRAINLIST_H


#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorSphereGermGrain : public COperator
{
public:
    OperatorSphereGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,Distribution * dist)
        {
            RandomGeometry::sphere(*in,*dist);
        }
    };

};
#endif // OPERATORSPHEREGRAINLIST_H
