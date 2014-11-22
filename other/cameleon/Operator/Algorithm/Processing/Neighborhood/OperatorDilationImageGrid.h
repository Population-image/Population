#ifndef OPERATORDILATIONMatN_H
#define OPERATORDILATIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"

#include"algorithm/Processing.h"
using namespace pop;
class OperatorDilationMatN : public COperator
{
public:
    OperatorDilationMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,pop::F64 radius,pop::F64 norm,  BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::dilation(*in1cast,radius,norm);
            out=outcast;
        }
    };

};

#endif // OPERATORDILATIONMatN_H
