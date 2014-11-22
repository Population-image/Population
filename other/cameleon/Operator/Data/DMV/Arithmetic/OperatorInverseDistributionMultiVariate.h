#ifndef OPERATORINVERSEDistributionMultiVariate_H
#define OPERATORINVERSEDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorInverseDistributionMultiVariate : public COperator
{
public:
    OperatorInverseDistributionMultiVariate();
    void exec();
    COperator * clone();
};


#endif // OPERATORINVERSEDistributionMultiVariate_H
