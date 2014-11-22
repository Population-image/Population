#ifndef OPERATORCONVERTFLOATMatN_H
#define OPERATORCONVERTFLOATMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertFloatMatN : public COperator
{
public:
    OperatorConvertFloatMatN();
    void exec();
    COperator * clone();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out){
            typedef typename FunctionTypeTraitsSubstituteF<Type,pop::F64>::Result TypeFloat;
            MatN<DIM,TypeFloat > * outcast = new MatN<DIM,TypeFloat>(in1cast->getDomain());
            *outcast = *in1cast;
            out =outcast;
        }
    };

};
#endif // OPERATORCONVERTFLOATMatN_H
