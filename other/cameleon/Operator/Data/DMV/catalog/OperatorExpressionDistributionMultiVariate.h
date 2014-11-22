#ifndef OPERATOREXPRESSIONDISTRIBUTIONMULTIVARIATE_H
#define OPERATOREXPRESSIONDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorExpressionDistributionMultiVariate : public COperator
{
public:
    OperatorExpressionDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATOREXPRESSIONDISTRIBUTIONMULTIVARIATE_H
