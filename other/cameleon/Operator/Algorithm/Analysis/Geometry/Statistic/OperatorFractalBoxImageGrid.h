#ifndef OPERATORFRACTALBOXMatN_H
#define OPERATORFRACTALBOXMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorFractalBoxMatN : public COperator
{
public:
    OperatorFractalBoxMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Mat2F64*& m){
            m  = new Mat2F64;
            *m = Analysis::fractalBox(*in1cast);
        }
    };

};
#endif // OPERATORFRACTALBOXMatN_H
