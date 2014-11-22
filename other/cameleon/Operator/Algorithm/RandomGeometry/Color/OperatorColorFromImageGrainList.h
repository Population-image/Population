#ifndef OPERATORCOLORFROMIMAGEGRAINLIST_H
#define OPERATORCOLORFROMIMAGEGRAINLIST_H
#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorColorFromImageGermGrain : public COperator
{
public:
    OperatorColorFromImageGermGrain();
    void exec();
    COperator * clone();

    struct foo
    {
        template<int DIM>
        void operator()(GermGrain<DIM> * in,MatN<DIM,RGBUI8 > * img)
        {
            RandomGeometry::RGBFromMatrix(*in,*img);
        }
    };

};
#endif // OPERATORCOLORFROMIMAGEGRAINLIST_H
