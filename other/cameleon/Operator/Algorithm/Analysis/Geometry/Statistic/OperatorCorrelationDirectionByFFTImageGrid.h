#ifndef OPERATORCorrelationDirectionByFFTDIRECTIONBYFFTMatN_H
#define OPERATORCorrelationDirectionByFFTDIRECTIONBYFFTMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorCorrelationDirectionByFFTMatN : public COperator
{
public:
    OperatorCorrelationDirectionByFFTMatN();
    void exec();
    COperator * clone();
    struct foo
    {

        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,BaseMatN * & fout ){

            MatN<DIM,pop::F64> * foutcast = new MatN<DIM,pop::F64>(in1cast->getDomain());
            * foutcast = Analysis::correlationDirectionByFFT(*in1cast);
            fout = foutcast;
        }
    };

};

#endif // OPERATORCorrelationDirectionByFFTDIRECTIONBYFFTMatN_H
