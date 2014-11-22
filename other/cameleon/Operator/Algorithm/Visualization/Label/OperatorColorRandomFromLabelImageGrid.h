#ifndef OPERATORCOLORRANDOMFROMLABELMatN_H
#define OPERATORCOLORRANDOMFROMLABELMatN_H

#include"COperator.h"
#include"Data/MatNCameleon/MatNN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorColorRandomFromLabelMatN : public COperator
{
public:
    OperatorColorRandomFromLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out)
        {

            MatNN<DIM,RGBUI8 > * outcast = new MatNN<DIM,RGBUI8 >(in1cast->getDomain());
            *outcast = Visualization::labelToRandomRGB(*in1cast);
            out=outcast;
        }
    };

};

#endif // OPERATORCOLORRANDOMFROMLABELMatN_H
