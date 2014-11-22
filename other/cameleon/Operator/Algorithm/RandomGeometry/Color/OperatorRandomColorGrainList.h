#ifndef OPERATORRANDOMCOLORGRAINLIST_H
#define OPERATORRANDOMCOLORGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorRandomColorGermGrain : public COperator
{
public:
    OperatorRandomColorGermGrain();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,Distribution * distred,Distribution * distgreen,Distribution * distblue)
        {
            DistributionMultiVariate dmultir(* distred);
            DistributionMultiVariate dmultig(* distgreen);
            DistributionMultiVariate dmultib(* distblue);
            DistributionMultiVariate dcoupled(dmultir,DistributionMultiVariate(dmultig,dmultib));
            RandomGeometry::RGBRandom(* in,dcoupled);
        }
    };

};
#endif // OPERATORRANDOMCOLORGRAINLIST_H
