#ifndef OPERATORSUBDistributionMultiVariate_H
#define OPERATORSUBDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorSubDistributionMultiVariate : public COperator
{
public:
    OperatorSubDistributionMultiVariate();
    void exec();
    COperator * clone();
};


#endif // OPERATORSUBDistributionMultiVariate_H
