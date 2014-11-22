#ifndef OPERATORINDEPENDANTDISTRIBUTIONMULTIVARIATE_H
#define OPERATORINDEPENDANTDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorIndependantDistributionMultiVariate : public COperator
{
public:
    OperatorIndependantDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORINDEPENDANTDISTRIBUTIONMULTIVARIATE_H
