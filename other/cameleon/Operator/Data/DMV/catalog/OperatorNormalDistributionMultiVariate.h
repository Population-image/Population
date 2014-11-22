#ifndef OPERATORNORMALDISTRIBUTIONMULTIVARIATE_H
#define OPERATORNORMALDISTRIBUTIONMULTIVARIATE_H
#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorNormalDistributionMultiVariate : public COperator
{
public:
    OperatorNormalDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORNORMALDISTRIBUTIONMULTIVARIATE_H
