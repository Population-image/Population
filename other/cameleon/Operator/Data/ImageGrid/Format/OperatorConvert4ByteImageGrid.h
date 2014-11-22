#ifndef OPERATORCONVERT4BYTEMatN_H
#define OPERATORCONVERT4BYTEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvert4ByteMatN : public COperator
{
public:
    OperatorConvert4ByteMatN();
    void exec();
    COperator * clone();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out){
            typedef typename FunctionTypeTraitsSubstituteF<Type,int>::Result Type4Byte;
            MatN<DIM,Type4Byte > * outcast = new MatN<DIM,Type4Byte>(in1cast->getDomain());
            *outcast = *in1cast;
            out =outcast;
        }
    };

};

#endif // OPERATORCONVERT4BYTEMatN_H
