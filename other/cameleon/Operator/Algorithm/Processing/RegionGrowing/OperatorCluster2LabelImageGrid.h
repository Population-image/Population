#ifndef OPERATORCLUSTER2LABELMatN_H
#define OPERATORCLUSTER2LABELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorCluster2LabelMatN : public COperator
{
public:
    OperatorCluster2LabelMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * binarycast,int norm,BaseMatN * &label)
        {
            MatN<DIM,UI32> * labelcast = new MatN<DIM,UI32>(binarycast->getDomain());

            *labelcast = Processing::clusterToLabel(*binarycast,  norm);
            label = labelcast;
        }
    };
};

#endif // OPERATORCLUSTER2LABELMatN_H
