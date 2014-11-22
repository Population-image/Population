#ifndef OPERATORTHRESHOLDMatN_H
#define OPERATORTHRESHOLDMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;

class OperatorThresholdMatN : public COperator
{
public:
    OperatorThresholdMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double v1,double v2, BaseMatN * &out)
        {
            MatN<DIM,unsigned char > * outcast = new MatN<DIM,unsigned char>(in1cast->getDomain());
            *outcast = Processing::threshold(* in1cast,v1,v2);
            out =outcast;

        }
    };

};

#endif // OPERATORTHRESHOLDMatN_H
