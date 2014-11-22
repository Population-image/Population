#ifndef OPERATORALTERNATESEQUENTIALOCMatN_H
#define OPERATORALTERNATESEQUENTIALOCMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"

#include"algorithm/Processing.h"
using namespace pop;
class OperatorAlternateSequentialOCMatN : public COperator
{
public:
    OperatorAlternateSequentialOCMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,pop::F64 radius,pop::F64 norm,  BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::alternateSequentialOC(*in1cast,radius,norm);
            out=outcast;
        }
    };

};

#endif // OPERATORALTERNATESEQUENTIALOCMatN_H
