#ifndef OPERATORADDDistributionMultiVariate_H
#define OPERATORADDDistributionMultiVariate_H
#include"COperator.h"
#include<DataDistributionMultiVariate.h>

class OperatorAddDistributionMultiVariate : public COperator
{
public:
    OperatorAddDistributionMultiVariate();
    void exec();
    COperator * clone();
};

#endif // OPERATORADDDistributionMultiVariate_H
