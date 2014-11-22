#ifndef OPERATORMAXDistributionMultiVariate_H
#define OPERATORMAXDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorMaxDistributionMultiVariate : public COperator
{
public:
    OperatorMaxDistributionMultiVariate();
    void exec();
    COperator * clone();
};
#endif // OPERATORMAXDistributionMultiVariate_H
