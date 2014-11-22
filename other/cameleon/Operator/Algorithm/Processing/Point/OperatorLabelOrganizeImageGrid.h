#ifndef OPERATORLABELORGANIZEMatN_H
#define OPERATORLABELORGANIZEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorLabelOrganizeMatN : public COperator
{
public:
    OperatorLabelOrganizeMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            * outcast = Processing::greylevelRemoveEmptyValue(* in1cast);
            out=outcast;
        }
    };

};

#endif // OPERATORLABELORGANIZEMatN_H
