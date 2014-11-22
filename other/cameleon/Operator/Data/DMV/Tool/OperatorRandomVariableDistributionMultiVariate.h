#ifndef OPERATORRANDOMVARIABLEDistributionMultiVariate_H
#define OPERATORRANDOMVARIABLEDistributionMultiVariate_H

#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorRandomVariableDistributionMultiVariate : public COperator
{
public:
    OperatorRandomVariableDistributionMultiVariate();
    void exec();
    COperator * clone();
};
#endif // OPERATORRANDOMVARIABLEDistributionMultiVariate_H
