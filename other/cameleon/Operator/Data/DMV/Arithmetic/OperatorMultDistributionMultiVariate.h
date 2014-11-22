#ifndef OPERATORMULTDistributionMultiVariate_H
#define OPERATORMULTDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorMultDistributionMultiVariate : public COperator
{
public:
    OperatorMultDistributionMultiVariate();
    void exec();
    COperator * clone();
};


#endif // OPERATORMULTDistributionMultiVariate_H
