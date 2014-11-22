#ifndef OPERATORBINARYFROMLABELSELECTIONMatN_H
#define OPERATORBINARYFROMLABELSELECTIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorBinaryFromLabelSelectionMatN : public COperator
{
public:
    OperatorBinaryFromLabelSelectionMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double v1,double v2, BaseMatN * &out)
        {
            MatN<DIM,unsigned char > * outcast = new MatN<DIM,unsigned char>(in1cast->getDomain());
            typename MatN<DIM,Type>::IteratorEDomain it(in1cast->getIteratorEDomain());
            FunctorThreshold<unsigned char,Type, Type> func(v1,v2-1);
            FunctionProcedureFunctorUnaryF(*in1cast,func,it,*outcast);
            out =outcast;

        }
    };

};

#endif // OPERATORBINARYFROMLABELSELECTIONMatN_H
