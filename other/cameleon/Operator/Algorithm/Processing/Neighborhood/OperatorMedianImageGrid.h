#ifndef OPERATORMEDIANMatN_H
#define OPERATORMEDIANMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorMedianMatN : public COperator
{
public:
    OperatorMedianMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,pop::F64 radius,pop::F64 norm,  BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::median(*in1cast,radius,norm);
            out=outcast;
        }
    };

};
#endif // OPERATORMEDIANMatN_H
