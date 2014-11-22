#ifndef OPERATORROTATIONMatN_H
#define OPERATORROTATIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
#include"algorithm/GeometricalTransformation.h"
using namespace  pop;
class OperatorRotationMatN : public COperator
{
public:
    OperatorRotationMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,Mat2F64* m,BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast =  new MatN<DIM,Type>(in->getDomain());
            *outcast = GeometricalTransformation::rotate(*in,* m,in->getDomain()*0.5);
            out = outcast;
        }
    };
};
#endif // OPERATORROTATIONMatN_H
