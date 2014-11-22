#ifndef OPERATORRHOMBOHEDRONGRAINLIST_H
#define OPERATORRHOMBOHEDRONGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorRhombohedronGermGrain : public COperator
{
public:
    OperatorRhombohedronGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        void operator()(GermGrain3 * in,Distribution * dist1, Distribution * dist2,Distribution * anglex,Distribution * angley,Distribution * anglez)
        {
            DistributionMultiVariate vangle(DistributionMultiVariate(*anglex,*angley),*anglez);
            RandomGeometry::rhombohedron(*in,*dist1, * dist2,vangle);
        }
    };

};

#endif // OPERATORRHOMBOHEDRONGRAINLIST_H
