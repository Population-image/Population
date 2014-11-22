#ifndef OPERATORDIFFUSIONCOEFFICIENTMatN_H
#define OPERATORDIFFUSIONCOEFFICIENTMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/PDE.h"
using namespace pop;
class OperatorDiffusionSelfCoefficientMatN : public COperator
{
public:
    OperatorDiffusionSelfCoefficientMatN();
    void exec();
    COperator * clone();
        struct foo
        {
            template<int DIM,typename Type>
            void operator()(MatN<DIM,Type> * in1cast,int nbrwalker,int timemax, Mat2F64*& m)
            {

                *m = PDE::randomWalk(* in1cast,nbrwalker,0.3,timemax);
            }
        };

};
#endif // OPERATORDIFFUSIONCOEFFICIENTMatN_H
