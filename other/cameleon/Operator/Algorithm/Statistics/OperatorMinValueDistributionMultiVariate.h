#ifndef OPERATORMINVALUEDISTRIBUTIONMULTIVARIATE_H
#define OPERATORMINVALUEDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"


class OperatorMinValueDistributionMultiVariate : public COperator
{
public:
    OperatorMinValueDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};

#endif // OPERATORMINVALUEDISTRIBUTIONMULTIVARIATE_H
