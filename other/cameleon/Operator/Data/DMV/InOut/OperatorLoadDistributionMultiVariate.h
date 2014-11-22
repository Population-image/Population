#ifndef OPERATORLOADDistributionMultiVariate_H
#define OPERATORLOADDistributionMultiVariate_H

#include"COperator.h"

class OperatorLoadDistributionMultiVariate: public COperator
{
public:
    OperatorLoadDistributionMultiVariate();
    void exec();
    COperator * clone();
};
#endif // OPERATORLOADDistributionMultiVariate_H
