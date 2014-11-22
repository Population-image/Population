#ifndef OPERATORDIVDistributionMultiVariate_H
#define OPERATORDIVDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorDivDistributionMultiVariate : public COperator
{
public:
    OperatorDivDistributionMultiVariate();
    void exec();
    COperator * clone();
};


#endif // OPERATORDIVDistributionMultiVariate_H
