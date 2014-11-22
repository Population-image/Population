#ifndef OPERATORINTEGRALDISTRIBUTIONMULTIVARIATE_H
#define OPERATORINTEGRALDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"


class OperatorIntegralDistributionMultiVariate : public COperator
{
public:
    OperatorIntegralDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};

#endif // OPERATORINTEGRALDISTRIBUTIONMULTIVARIATE_H
