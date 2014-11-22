#ifndef OPERATORFFTMatN_H
#define OPERATORFFTMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"

#include"algorithm/Processing.h"
#include"algorithm/Representation.h"
using namespace pop;
class OperatorFFTMatN : public COperator
{
public:
    OperatorFFTMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(MatN<DIM,ComplexF64 > * in1cast, BaseMatN * &out)
        {

            MatN<DIM,ComplexF64 > * outcast = new MatN<DIM,ComplexF64 >(in1cast->getDomain());

            MatN<DIM,ComplexF64 >  temp;

            temp =Representation::truncateMulitple2(  * in1cast );
            *outcast =Representation::FFT(temp,1);
            out =outcast;
        }
    };

};
#endif // OPERATORFFTMatN_H
