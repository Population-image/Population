#ifndef OPERATORGETSIZEMatN_H
#define OPERATORGETSIZEMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
using namespace pop;
class OperatorGetSizeMatN : public COperator
{
public:
    OperatorGetSizeMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,VecF64  & p)
        {
            VecN<DIM,pop::F64> x;
            x= in->getDomain();
            p= x;
        }
    };

};

#endif // OPERATORGETSIZEMatN_H
