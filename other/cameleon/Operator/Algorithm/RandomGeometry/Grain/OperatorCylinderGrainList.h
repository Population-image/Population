#ifndef OPERATORCYLINDERGRAINLIST_H
#define OPERATORCYLINDERGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorCylinderGermGrain : public COperator
{
public:
    OperatorCylinderGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {

        void operator()(GermGrain3 * in,Distribution * dist1, Distribution * dist2,Distribution * anglex,Distribution * angley,Distribution * anglez)
        {
            DistributionMultiVariate vangle(DistributionMultiVariate(*anglex,*angley),*anglez);
             RandomGeometry::cylinder(*in,* dist1,* dist2,vangle);
        }
    };

};
#endif // OPERATORCYLINDERGRAINLIST_H
