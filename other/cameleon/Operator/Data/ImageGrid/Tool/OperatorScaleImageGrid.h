#ifndef OPERATORSCALEMatN_H
#define OPERATORSCALEMatN_H


#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/GeometricalTransformation.h"
using namespace pop;
class OperatorScaleMatN : public COperator
{
public:
    OperatorScaleMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in,double lambdax, double lambday,double , BaseMatN * &out)
        {
            MatN<2,Type> * outcast = new MatN<2,Type>();

            * outcast = GeometricalTransformation::scale(*in,Vec2I32(in->getDomain()(0)*lambdax,in->getDomain()(1)*lambday));
            out = outcast;
        }
        template<typename Type>
        void operator()(MatN<3,Type> * in,double lambdax, double lambday,double lambdaz, BaseMatN * &out)
        {
            MatN<3,Type> * outcast = new MatN<3,Type>(*in);
            * outcast = GeometricalTransformation::scale(*in,Vec3I32(in->getDomain()(0)*lambdax,in->getDomain()(1)*lambday,in->getDomain()(2)*lambdaz));
            out = outcast;
        }
    };
};

#endif // OPERATORSCALEMatN_H
