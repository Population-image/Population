#ifndef OPERATORMULTSCALARMatN_H
#define OPERATORMULTSCALARMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorMultScalarMatN : public COperator
{
public:
    OperatorMultScalarMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double value, BaseMatN * &out)throw(pexception)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            FunctorValueAfter<Type,double,Type,MultiplicationF2<Type,double,Type> > op(value);
            std::transform (in1cast->data(), in1cast->data()+in1cast->getDomain().multCoordinate(), outcast->data(),  op);
            out=outcast;


        }
    };

};

#endif // OPERATORMULTSCALARMatN_H
