#ifndef OPERATORCOUPLEDDISTRIBUTIONMULTIVARIATE_H
#define OPERATORCOUPLEDDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorCoupledDistributionMultiVariate : public COperator
{
public:
    OperatorCoupledDistributionMultiVariate();
    void exec();
    COperator * clone();
};
#endif // OPERATORCOUPLEDDISTRIBUTIONMULTIVARIATE_H
