#ifndef OPERATORDIVSCALARMatN_H
#define OPERATORDIVSCALARMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorDivScalarMatN : public COperator
{
public:
    OperatorDivScalarMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double value, BaseMatN * &out)throw(pexception)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            MatN<DIM,Type> temp(in1cast->getDomain(),value);
            (*outcast) =   temp.divTermByTerm(*in1cast);
            out=outcast;


        }
    };

};

#endif // OPERATORDIVSCALARMatN_H
