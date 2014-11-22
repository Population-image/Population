#ifndef OPERATOREULERPOINCAREMatN_H
#define OPERATOREULERPOINCAREMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorEulerPoincareMatN : public COperator
{
public:
    OperatorEulerPoincareMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,string file,double &value)
        {
            value  = Analysis::eulerPoincare(* in1cast,file.c_str());
        }
    };

};

#endif // OPERATOREULERPOINCAREMatN_H
