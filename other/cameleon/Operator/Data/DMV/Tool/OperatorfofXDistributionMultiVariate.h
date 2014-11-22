#ifndef OPERATORFOFXDistributionMultiVariate_H
#define OPERATORFOFXDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorfofxDistributionMultiVariate : public COperator
{
public:
    OperatorfofxDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORFOFXDistributionMultiVariate_H
