#ifndef OPERATORHISTOGRAMMatN_H
#define OPERATORHISTOGRAMMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;

class OperatorHistogramMatN : public COperator
{
public:
    OperatorHistogramMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Mat2F64*& m){
            m  = new Mat2F64;
            *m  = Analysis::histogram(* in1cast);
        }
    };

};

#endif // OPERATORHISTOGRAMMatN_H
