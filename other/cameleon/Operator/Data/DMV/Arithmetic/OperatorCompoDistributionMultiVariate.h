#ifndef OPERATORCOMPODistributionMultiVariate_H
#define OPERATORCOMPODistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorCompositionDistributionMultiVariate : public COperator
{
public:
    OperatorCompositionDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORCOMPODistributionMultiVariate_H
