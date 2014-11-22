#ifndef OPERATORMAXVALUEMatN_H
#define OPERATORMAXVALUEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorMaxValueMatN : public COperator
{
public:
    OperatorMaxValueMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double &value)
        {         
            Type _value = Analysis::maxValue(* in1cast);
            value=static_cast<double>(_value);
        }
    };

};

#endif // OPERATORMAXVALUEMatN_H
