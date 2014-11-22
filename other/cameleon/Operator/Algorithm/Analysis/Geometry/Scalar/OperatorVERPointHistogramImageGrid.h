#ifndef OPERATORVERPOINTHistogramMatN_H
#define OPERATORVERPOINTHistogramMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;

class OperatorVERPointHistogramMatN : public COperator
{
public:
    OperatorVERPointHistogramMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,VecF64  x, int rmax, int norm,Mat2F64*& m)throw(pexception)
        {
            m  = new Mat2F64;
            VecN<DIM,pop::F64> xx;
            if(x.size()==0)
                xx=in1cast->getDomain()*0.5;
            else{
                try{
                    xx=x;
                }catch(string){
                    throw(pexception("Your input point must have the same dimension as the image"));
                }
            }
            *m  = Analysis::VERVecNHistogram(*in1cast,xx, rmax,norm);
        }
    };

};

#endif // OPERATORVERPOINTHistogramMatN_H
