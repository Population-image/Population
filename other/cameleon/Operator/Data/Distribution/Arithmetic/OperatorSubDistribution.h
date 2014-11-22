#ifndef OPERATORSUBDISTRIBUTION_H
#define OPERATORSUBDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorSubDistribution : public COperator
{
public:
    OperatorSubDistribution();
    void exec();
    COperator * clone();
};


#endif // OPERATORSUBDISTRIBUTION_H
