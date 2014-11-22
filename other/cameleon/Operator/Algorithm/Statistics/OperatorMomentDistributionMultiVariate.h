#ifndef OPERATORMOMENTDISTRIBUTIONMULTIVARIATE_H
#define OPERATORMOMENTDISTRIBUTIONMULTIVARIATE_H

#include<DataDistributionMultiVariate.h>

class OperatorMomentDistributionMultiVariate : public COperator
{
public:
    OperatorMomentDistributionMultiVariate();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORMOMENTDISTRIBUTIONMULTIVARIATE_H
