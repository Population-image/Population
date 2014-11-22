#ifndef OPERATORLABELADDMatN_H
#define OPERATORLABELADDMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
#include"algorithm/Processing.h"
using namespace pop;

class OperatorLabelAddMatN : public COperator
{
public:
    OperatorLabelAddMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,BaseMatN * in2, BaseMatN * &out)throw(pexception)
        {
            if(MatN<DIM,Type> * in2cast = dynamic_cast<MatN<DIM,Type> *>(in2)){

                MatN<DIM,Type> *outcast = new MatN<DIM,Type>(in1cast->getDomain());

                *outcast = Processing::labelMerge(* in1cast,* in2cast)  ;
                out=outcast;
            }


        }
    };

};
#endif // OPERATORLABELADDMatN_H
