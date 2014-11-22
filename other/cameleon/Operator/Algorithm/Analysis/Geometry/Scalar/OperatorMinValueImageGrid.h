#ifndef OPERATORMINVALUEMatN_H
#define OPERATORMINVALUEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorMinValueMatN : public COperator
{
public:
    OperatorMinValueMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double &value)
        {
            Type _value = Analysis::minValue(* in1cast);
            value=static_cast<double>(_value);
        }
    };

};

#endif // OPERATORMINVALUEMatN_H
