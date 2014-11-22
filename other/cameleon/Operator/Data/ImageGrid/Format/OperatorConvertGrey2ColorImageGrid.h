#ifndef OPERATORCONVERTGREY2COLORMatN_H
#define OPERATORCONVERTGREY2COLORMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertGrey2ColorMatN : public COperator
{
public:
    OperatorConvertGrey2ColorMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(MatN<DIM,unsigned char > * in1cast, BaseMatN * &out)
        {

            MatN<DIM,RGBUI8 > * outcast = new MatN<DIM,RGBUI8 >(in1cast->getDomain());
            *outcast=* in1cast;
            out =outcast;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, BaseMatN * &out)
        {
            MatN<DIM,RGB<Type> > * outcast = new MatN<DIM,RGB<Type> >(*in1cast);
            out =outcast;
        }
    };

};


#endif // OPERATORCONVERTGREY2COLORMatN_H
