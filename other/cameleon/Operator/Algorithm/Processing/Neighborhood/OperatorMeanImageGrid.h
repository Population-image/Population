#ifndef OPERATORMEANMatN_H
#define OPERATORMEANMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorMeanMatN : public COperator
{
public:
    OperatorMeanMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,pop::F64 radius,pop::F64 norm,  BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::mean(*in1cast,radius,norm);
            out=outcast;
        }
    };

};

#endif // OPERATORMEANMatN_H
