#ifndef OPERATORVOLUMEFRACTIONMatN_H
#define OPERATORVOLUMEFRACTIONMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorAreaMatN : public COperator
{
public:
    OperatorAreaMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Mat2F64*& m){
            m  = new Mat2F64;
            *m  = Analysis::area(* in1cast);
        }
    };

};

#endif // OPERATORVOLUMEFRACTIONMatN_H
