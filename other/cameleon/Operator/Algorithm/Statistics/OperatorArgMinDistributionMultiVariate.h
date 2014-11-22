#ifndef OPERATORARGMINDISTRIBUTIONMULTIVARIATE_H
#define OPERATORARGMINDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"


class OperatorArgMinDistributionMultiVariate : public COperator
{
public:
    OperatorArgMinDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};

#endif // OPERATORARGMINDISTRIBUTIONMULTIVARIATE_H
