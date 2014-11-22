#ifndef OPERATORSMOOTHDERICHEMatN_H
#define OPERATORSMOOTHDERICHEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorSmoothDericheMatN : public COperator
{
public:
    OperatorSmoothDericheMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double alpha, BaseMatN * &out)
        {
            MatN<DIM,Type>* outcast  =  new  MatN<DIM,Type>(in1cast->getDomain());
            * outcast = Processing::smoothDeriche(*in1cast,alpha);
            out=outcast;
        }
    };

};

#endif // OPERATORSMOOTHDERICHEMatN_H
