#ifndef OPERATORCORRELATIONMatN_H
#define OPERATORCORRELATIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorCorrelationMatN : public COperator
{
public:
    OperatorCorrelationMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int number,int length,Mat2F64*& m){
            m  = new Mat2F64;

            *m = Analysis::correlation(*in1cast,length,number);
        }
    };

};

#endif // OPERATORCORRELATIONMatN_H
