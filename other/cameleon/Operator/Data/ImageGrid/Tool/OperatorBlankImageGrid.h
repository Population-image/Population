#ifndef OPERATORBLANKMatN_H
#define OPERATORBLANKMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorBlankMatN : public COperator
{
public:
    OperatorBlankMatN();
    void exec();
    void initState();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * & h,VecF64  v)
        {
            VecN<DIM,pop::F64> p;
            p=v;
            h->resize(p);
        }
    };
};

#endif // OPERATORBLANKMatN_H
