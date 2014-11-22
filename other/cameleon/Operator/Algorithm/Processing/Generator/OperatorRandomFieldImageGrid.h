#ifndef OPERATORRANDOMVARIABLEMatN_H
#define OPERATORRANDOMVARIABLEMatN_H


#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
#include"algorithm/Processing.h"
class OperatorRandomFieldMatN : public COperator
{
public:
    OperatorRandomFieldMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Distribution & f, BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            Processing::randomField(in1cast->getDomain(),f,*outcast);
             out = outcast;

        }
    };

};
#endif // OPERATORRANDOMVARIABLEMatN_H
