#ifndef OPERATORINNERPRODUCTDISTRIBUTION_H
#define OPERATORINNERPRODUCTDISTRIBUTION_H

#include<DataDistribution.h>

class OperatorInnerProductDistribution : public COperator
{
public:
    OperatorInnerProductDistribution();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORINNERPRODUCTDISTRIBUTION_H
