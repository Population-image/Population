#ifndef OPERATORLDISTANCEMatN_H
#define OPERATORLDISTANCEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorLDistanceMatN : public COperator
{
public:
    OperatorLDistanceMatN();
    void exec();
    COperator * clone();
        void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm,Mat2F64*& m,BaseMatN * &out){
            m  = new Mat2F64;
            MatN<DIM,unsigned char> * outcast = new MatN<DIM,unsigned char>(in1cast->getDomain());
            *m = Analysis::ldistance(  *in1cast,norm,* outcast);
            out=outcast;
        }
    };

};

#endif // OPERATORLDISTANCEMatN_H
