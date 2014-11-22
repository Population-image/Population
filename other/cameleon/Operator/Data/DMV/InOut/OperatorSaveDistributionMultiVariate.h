#ifndef OPERATORSAVEDistributionMultiVariate_H
#define OPERATORSAVEDistributionMultiVariate_H

#include"COperator.h"

class OperatorSaveDistributionMultiVariate: public COperator
{
public:
    OperatorSaveDistributionMultiVariate();
    void exec();
    COperator * clone();
};
#endif // OPERATORSAVEDistributionMultiVariate_H
