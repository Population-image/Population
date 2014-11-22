#ifndef OPERATORSETPOINTMatN_H
#define OPERATORSETPOINTMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorSetPointMatN : public COperator
{
public:

    OperatorSetPointMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, VecF64  x,double r, double g,  double b)
        {
            VecN<DIM,pop::F64> p;
            p=x;
            in1cast->operator ()(p).r()=r;
            in1cast->operator ()(p).g()=g;
            in1cast->operator ()(p).b()=b;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, VecF64  x,double scalar)
        {
            VecN<DIM,pop::F64> p;
            p=x;
            in1cast->operator ()(p) =  scalar;
        }
    };
};

#endif // OPERATORSETPOINTMatN_H
