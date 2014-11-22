#ifndef OPERATORCONSTMatN_H
#define OPERATORCONSTMatN_H


#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorConstMatN : public COperator
{
public:
    OperatorConstMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double value, BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(*in1cast);
            outcast->fill(value);
            out=outcast;

        }
    };

};
#endif // OPERATORCONSTMatN_H
