#ifndef OPERATORRANDOMBLACKORWHITEGRAINLIST_H
#define OPERATORRANDOMBLACKORWHITEGRAINLIST_H

#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorRandomBlackOrWhiteGermGrain : public COperator
{
public:
    OperatorRandomBlackOrWhiteGermGrain();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in)
        {
            RandomGeometry::RGBRandomBlackOrWhite(*in);
        }
    };

};

#endif // OPERATORRANDOMBLACKORWHITEGRAINLIST_H
