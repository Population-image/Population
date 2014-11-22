#ifndef OPERATORCHORDMatN_H
#define OPERATORCHORDMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorChordMatN : public COperator
{
public:
    OperatorChordMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int number,Mat2F64*& m){
            m  = new Mat2F64;
            *m = Analysis::chord(*in1cast,number);
        }
    };

};

#endif // OPERATORCHORDMatN_H
