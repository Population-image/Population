#ifndef OPERATORGETPOINTMatN_H
#define OPERATORGETPOINTMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorGetPointMatN : public COperator
{
public:

    OperatorGetPointMatN();
    void exec();
    COperator * clone();
    virtual void updateMarkingAfterExecution();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, VecF64  x,double &r, double &g,  double &b)
        {
            VecN<DIM,pop::F64> p;
            p=x;
            r = in1cast->operator ()(p).r();
            g = in1cast->operator ()(p).g();
            b = in1cast->operator ()(p).b();
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, VecF64  x,double &scalar)
        {
            VecN<DIM,pop::F64> p;
            p=x;
            scalar = in1cast->operator ()(p);
        }
    };
private:
    bool _color;
};

#endif // OPERATORGETPOINTMatN_H
