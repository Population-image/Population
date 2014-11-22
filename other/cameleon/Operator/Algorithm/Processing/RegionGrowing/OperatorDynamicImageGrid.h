#ifndef OPERATORDYNAMICMatN_H
#define OPERATORDYNAMICMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorDynamicMatN : public COperator
{
public:
    OperatorDynamicMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * fcast,int num, int norm,BaseMatN * &h)throw(pexception)
        {

            MatN<DIM,Type> * hcast =  new MatN<DIM,Type>;
            *hcast = Processing::dynamic(*fcast,num,norm);
            h=hcast;
        }
    };

};

#endif // OPERATORDYNAMICMatN_H
