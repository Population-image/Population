#ifndef OPERATORRESIZEMatN_H
#define OPERATORRESIZEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
using namespace pop;
class OperatorResizeMatN : public COperator
{
public:
    OperatorResizeMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,VecF64  p,BaseMatN *& out)throw(pexception)
        {
            VecN<DIM,pop::F64> x;
            x=p;
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(*in);
            outcast->resize(x);
            out = outcast;
        }
    };

};


#endif // OPERATORRESIZEMatN_H
