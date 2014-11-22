#ifndef OPERATORADDSCALARMatN_H
#define OPERATORADDSCALARMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorAddScalarMatN : public COperator
{
public:
    OperatorAddScalarMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double value, BaseMatN * &out)throw(pexception)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            MatN<DIM,Type> temp(in1cast->getDomain(),value);
            (*outcast) =   temp+(*in1cast);
            out=outcast;


        }
    };

};
#endif // OPERATORADDSCALARMatN_H
