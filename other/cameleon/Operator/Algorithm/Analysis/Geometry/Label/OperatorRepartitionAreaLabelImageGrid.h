#ifndef OPERATORREPARTITIONAREALABELMatN_H
#define OPERATORREPARTITIONAREALABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorRepartitionAreaLabelMatN : public COperator
{
public:
    OperatorRepartitionAreaLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,VecF64  & v){
            v = Analysis::areaByLabel(*in1cast);
        }
    };

};
#endif // OPERATORREPARTITIONAREALABELMatN_H
