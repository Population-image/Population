#ifndef OPERATORMINUXMatN_H
#define OPERATORMINUXMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorMinusMatN : public COperator
{
public:
    OperatorMinusMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * &out)
        {

                MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
                MatN<DIM,Type> temp(outcast->getDomain(),numeric_limits<Type>::max());
                (*outcast) =   temp-(*in1cast);
                out=outcast;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double value, BaseMatN * &out)
        {

                MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
                MatN<DIM,Type> temp(outcast->getDomain(),value);
                (*outcast) =   temp-(*in1cast);
                out=outcast;
        }
    };


};

#endif // OPERATORMINUXMatN_H
