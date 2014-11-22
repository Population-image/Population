#ifndef OPERATORTRANSLATIONMatN_H
#define OPERATORTRANSLATIONMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
#include"algorithm/GeometricalTransformation.h"
using namespace pop;
class OperatorTranslationMatN : public COperator
{
public:
    OperatorTranslationMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,VecF64  t,BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast =  new MatN<DIM,Type>();
            VecN<DIM,pop::F64> p;
            p=t;
            *outcast= GeometricalTransformation::translate(* in,p);
            out = outcast;
        }
    };
};
#endif // OPERATORTRANSLATIONMatN_H
