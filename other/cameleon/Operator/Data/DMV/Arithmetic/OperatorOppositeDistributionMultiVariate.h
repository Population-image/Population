#ifndef OPERATOROPPOSITEDistributionMultiVariate_H
#define OPERATOROPPOSITEDistributionMultiVariate_H
#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorOppositeDistributionMultiVariate : public COperator
{
public:
    OperatorOppositeDistributionMultiVariate();
    void exec();
    COperator * clone();
};


#endif // OPERATOROPPOSITEDistributionMultiVariate_H
