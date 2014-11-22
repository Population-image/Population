#ifndef OPERATORREPARTITIONPERIMETERLABELMatN_H
#define OPERATORREPARTITIONPERIMETERLABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorRepartitionPerimeterLabelMatN : public COperator
{
public:
    OperatorRepartitionPerimeterLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,VecF64  &v){
            v = Analysis::perimeterByLabel(*in1cast);
        }
    };

};

#endif // OPERATORREPARTITIONPERIMETERLABELMatN_H
