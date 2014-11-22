#ifndef OPERATORCONVERT1BYTEMatN_H
#define OPERATORCONVERT1BYTEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvert1ByteMatN : public COperator
{
public:
    OperatorConvert1ByteMatN();
    void exec();
    COperator * clone();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out){
            typedef typename FunctionTypeTraitsSubstituteF<Type,pop::UI8>::Result TypeUC;
            MatN<DIM,TypeUC > * outcast = new MatN<DIM,TypeUC>(in1cast->getDomain());
            *outcast = *in1cast;
//            FunctionProcedureConvertF2Any<MatN<DIM,Type>,MatN<DIM,TypeUC>,ArithmeticSaturation >(,;
            out =outcast;
        }        
    };

};

#endif // OPERATORCONVERT1BYTEMatN_H
