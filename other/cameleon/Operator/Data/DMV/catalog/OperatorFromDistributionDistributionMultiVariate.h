#ifndef OPERATORFROMDISTRIBUTIONDISTRIBUTIONMULTIVARIATE_H
#define OPERATORFROMDISTRIBUTIONDISTRIBUTIONMULTIVARIATE_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorFromDistributionDistributionMultiVariate : public COperator
{
public:
    OperatorFromDistributionDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORFROMDISTRIBUTIONDISTRIBUTIONMULTIVARIATE_H
