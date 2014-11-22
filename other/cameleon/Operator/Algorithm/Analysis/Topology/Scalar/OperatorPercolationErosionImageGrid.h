#ifndef OPERATORPERCOLATIONEROSIONMatN_H
#define OPERATORPERCOLATIONEROSIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorPercolationErosionMatN : public COperator
{
public:
    OperatorPercolationErosionMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm, Mat2F64*& m){
            m  = new Mat2F64;
            *m = Analysis::percolationErosion(  *in1cast,norm);
        }
    };

};
#endif // OPERATORPERCOLATIONEROSIONMatN_H
