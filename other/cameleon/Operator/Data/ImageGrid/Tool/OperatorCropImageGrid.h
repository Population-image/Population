#ifndef OPERATORCROPMatN_H
#define OPERATORCROPMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
using namespace pop;
class OperatorCropMatN : public COperator
{
public:
    OperatorCropMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in,VecF64  xmin,VecF64  xmax, BaseMatN * &out)throw(pexception)
        {
            MatN<2,Type> * outcast =  new MatN<2,Type>();
            VecN<2,pop::F64> pmin,pmax;
            pmin=xmin;
            pmax=xmax;
            *outcast = (* in)(pmin,pmax);
            out = outcast;
        }

        template<typename Type>
        void operator()(MatN<3,Type> * in,VecF64  xmin,VecF64  xmax, BaseMatN * &out)throw(pexception)
        {
            MatN<3,Type> * outcast =  new MatN<3,Type>();
            VecN<3,pop::F64> pmin,pmax;
            pmin=xmin;
            pmax=xmax;
            *outcast = (* in)(pmin,pmax);
            out = outcast;
        }
    };


};

#endif // OPERATORCROPMatN_H
