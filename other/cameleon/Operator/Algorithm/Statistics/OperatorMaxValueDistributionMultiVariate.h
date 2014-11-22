#ifndef OPERATORMAXVALUEDISTRIBUTIONMULTIVARIATE_H
#define OPERATORMAXVALUEDISTRIBUTIONMULTIVARIATE_H


#include"COperator.h"


class OperatorMaxValueDistributionMultiVariate : public COperator
{
public:
    OperatorMaxValueDistributionMultiVariate();
    void exec();
    COperator * clone();
        void initState();
};

#endif // OPERATORMAXVALUEDISTRIBUTIONMULTIVARIATE_H
