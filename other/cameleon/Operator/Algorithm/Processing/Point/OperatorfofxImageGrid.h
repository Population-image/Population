#ifndef OPERATORFOFXMatN_H
#define OPERATORFOFXMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorfofxMatN : public COperator
{
public:
    OperatorfofxMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Distribution * f, BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::fofx(* in1cast,*f);
            out = outcast;
        }
    };

};

#endif // OPERATORFOFXMatN_H
