#ifndef OPERATORREPARTITIONCONTACTLABELMatN_H
#define OPERATORREPARTITIONCONTACTLABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorRepartitionPerimeterContactBetweenLabelMatN : public COperator
{
public:
    OperatorRepartitionPerimeterContactBetweenLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,VecF64 & v){
            v =  Analysis::perimeterContactBetweenLabel(*in1cast);
        }
    };

};
#endif // OPERATORREPARTITIONCONTACTLABELMatN_H
