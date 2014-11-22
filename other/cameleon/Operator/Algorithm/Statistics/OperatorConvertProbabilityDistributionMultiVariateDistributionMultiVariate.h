#ifndef OPERATORCONVERTPROBABILITYDISTRIBUTIONMULTIVARIATEDISTRIBUTIONMULTIVARIATE_H
#define OPERATORCONVERTPROBABILITYDISTRIBUTIONMULTIVARIATEDISTRIBUTIONMULTIVARIATE_H


#include"COperator.h"


class OperatorConvertProbabilityDistributionMultiVariateDistributionMultiVariate : public COperator
{
public:
    OperatorConvertProbabilityDistributionMultiVariateDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORCONVERTPROBABILITYDISTRIBUTIONMULTIVARIATEDISTRIBUTIONMULTIVARIATE_H
