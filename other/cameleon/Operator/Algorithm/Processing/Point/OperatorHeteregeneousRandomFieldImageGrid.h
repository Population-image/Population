#ifndef OPERATORHETEREGENEOUSRANDOMFIELDMatN_H
#define OPERATORHETEREGENEOUSRANDOMFIELDMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorHeteregeneousRandomFieldMatN : public COperator
{
public:
    OperatorHeteregeneousRandomFieldMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Distribution & f, BaseMatN * &out)
        {

             MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
//             *outcast = Processing::randomFieldHeteregeneous(* in1cast,f);
             out = outcast;

        }
    };

};

#endif // OPERATORHETEREGENEOUSRANDOMFIELDMatN_H
