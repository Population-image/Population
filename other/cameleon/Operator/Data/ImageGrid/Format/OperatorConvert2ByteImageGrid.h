#ifndef OPERATORCONVERT2BYTEMatN_H
#define OPERATORCONVERT2BYTEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvert2ByteMatN : public COperator
{
public:
    OperatorConvert2ByteMatN();
    void exec();
    COperator * clone();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out){
            typedef typename FunctionTypeTraitsSubstituteF<Type,pop::UI16>::Result TypeUC;
            MatN<DIM,TypeUC > * outcast = new MatN<DIM,TypeUC>(in1cast->getDomain());
            *outcast = *in1cast;
            out =outcast;
        }
    };

};

#endif // OPERATORCONVERT2BYTEMatN_H
