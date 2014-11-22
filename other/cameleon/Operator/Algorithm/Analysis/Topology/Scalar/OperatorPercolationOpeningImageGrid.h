#ifndef OPERATORPERCOLATIONOPENINGMatN_H
#define OPERATORPERCOLATIONOPENINGMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorPercolationOpeningMatN : public COperator
{
public:
    OperatorPercolationOpeningMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm, Mat2F64*& m){
            m  = new Mat2F64;
            *m = Analysis::percolationOpening(  *in1cast,norm);
        }
    };

};

#endif // OPERATORPERCOLATIONOPENINGMatN_H
