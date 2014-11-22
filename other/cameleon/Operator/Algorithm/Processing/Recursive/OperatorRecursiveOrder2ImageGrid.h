#ifndef OPERATORRECURSIVEORDER2MatN_H
#define OPERATORRECURSIVEORDER2MatN_H

#include<COperator.h>
#include"algorithm/Processing.h"
using namespace pop;
class OperatorRecursiveOrder2MatN : public COperator
{
public:
    OperatorRecursiveOrder2MatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, FunctorFilterRecursiveOrder2 & func , double c   , double w   , BaseMatN * &out)
        {
            typedef typename  FunctionTypeTraitsSubstituteF<typename MatN<DIM,Type>::F,pop::F64>::Result TypeFloat64;
            MatN<DIM,TypeFloat64 > * outcast = new MatN<DIM,TypeFloat64>(in1cast->getDomain());

            * outcast = Processing::recursive(* in1cast,func,c,w);


            out =outcast;

        }
    };

};

#endif // OPERATORRECURSIVEORDER2MatN_H
