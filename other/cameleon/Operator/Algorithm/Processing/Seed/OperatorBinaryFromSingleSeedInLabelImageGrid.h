#ifndef OPERATORBINARYFROMSEEDINLABELMatN_H
#define OPERATORBINARYFROMSEEDINLABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorBinaryFromSingleSeedInLabelMatN : public COperator
{
public:
    OperatorBinaryFromSingleSeedInLabelMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * labelcast,BaseMatN * bin, BaseMatN * &h)throw(pexception)
        {
            if(MatN<DIM,pop::UI8> * bincast = dynamic_cast<MatN<DIM,pop::UI8> *>(bin))
            {

                MatN<DIM,pop::UI8> *hcast = new MatN<DIM,pop::UI8>(labelcast->getDomain());
                *hcast = Processing::labelFromSingleSeed(* labelcast,*bincast);
                h=hcast;
            }
            else
            {
                throw(pexception("Input image must have the same type"));
            };

        }

    };


};

#endif // OPERATORBINARYFROMSEEDINLABELMatN_H
