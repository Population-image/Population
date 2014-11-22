#ifndef OPERATORARGMAXDISTRIBUTIONMULTIVARIATE_H
#define OPERATORARGMAXDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"


class OperatorArgMaxDistributionMultiVariate : public COperator
{
public:
    OperatorArgMaxDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORARGMAXDISTRIBUTIONMULTIVARIATE_H
