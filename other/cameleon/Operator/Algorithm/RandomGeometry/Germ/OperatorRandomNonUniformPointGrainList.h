#ifndef OPERATORRANDOMNONUNIFORMPOINTGRAINLIST_H
#define OPERATORRANDOMNONUNIFORMPOINTGRAINLIST_H

#include"COperator.h"
#include <DataImageGrid.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorRandomNonUniformPointGermGrain : public COperator
{
public:
    OperatorRandomNonUniformPointGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,GermGrainMother * &out)
        {

            GermGrain<DIM> * grain= new GermGrain<DIM>;
            * grain= RandomGeometry::poissonPointProcess(*in1cast);
            out = grain;
        }
    };
};

#endif // OPERATORRANDOMNONUNIFORMPOINTGRAINLIST_H
