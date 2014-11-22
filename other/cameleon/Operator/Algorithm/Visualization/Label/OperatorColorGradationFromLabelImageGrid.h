#ifndef OPERATORCOLORGRADATIONFROMLABELMatN_H
#define OPERATORCOLORGRADATIONFROMLABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorColorGradationFromLabelMatN : public COperator
{
public:
    OperatorColorGradationFromLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out)
        {

            MatN<DIM,RGBUI8 > * outcast = new MatN<DIM,RGBUI8 >(in1cast->getDomain());
            *outcast = Visualization::labelToRGBGradation(*in1cast);
            out=outcast;
        }
    };

};


#endif // OPERATORCOLORGRADATIONFROMLABELMatN_H
