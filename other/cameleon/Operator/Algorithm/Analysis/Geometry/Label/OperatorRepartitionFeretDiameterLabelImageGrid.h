#ifndef OPERATORREPARTITIONFERETDIAMETERLABELMatN_H
#define OPERATORREPARTITIONFERETDIAMETERLABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorRepartitionFeretDiameterLabelMatN : public COperator
{
public:
    OperatorRepartitionFeretDiameterLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, int norm , VecF64  &v){


            v = Analysis::feretDiameterByLabel(*in1cast,norm);
        }
    };

};

#endif // OPERATORREPARTITIONFERETDIAMETERLABELMatN_H
