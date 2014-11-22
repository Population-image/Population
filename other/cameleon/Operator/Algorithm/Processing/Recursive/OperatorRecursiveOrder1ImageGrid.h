#ifndef OPERATORRECURSIVEORDER1MatN_H
#define OPERATORRECURSIVEORDER1MatN_H

#include<COperator.h>
#include"algorithm/Processing.h"
using namespace pop;
class OperatorRecursiveOrder1MatN : public COperator
{
public:
    OperatorRecursiveOrder1MatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, FunctorFilterRecursiveOrder1 & func , double c   , double w   , BaseMatN * &out)
        {
            typedef typename  FunctionTypeTraitsSubstituteF<typename MatN<DIM,Type>::F,pop::F64>::Result Typefloat64;
            MatN<DIM,Typefloat64 > * outcast = new MatN<DIM,Typefloat64>(in1cast->getDomain());

            * outcast = Processing::recursive(* in1cast,func,c,w);

            out =outcast;

        }
    };

};
#endif // OPERATORRECURSIVEORDER1MatN_H
