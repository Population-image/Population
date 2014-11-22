#ifndef OPERATORSOBELMatN_H
#define OPERATORSOBELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorGradientNormSobelMatN : public COperator
{
public:
    OperatorGradientNormSobelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out)
        {
            MatN<DIM,Type>* outcast  =  new  MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::gradientMagnitudeSobel(*in1cast);
            out=outcast;
        }
    };

};
#endif // OPERATORSOBELMatN_H
