#ifndef OPERATORCONVERTCOLOR2GREYMatN_H
#define OPERATORCONVERTCOLOR2GREYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertColor2GreyMatN : public COperator
{
public:
    OperatorConvertColor2GreyMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = *in1cast;
            out =outcast;
        }
    };

};


#endif // OPERATORCONVERTCOLOR2GREYMatN_H
