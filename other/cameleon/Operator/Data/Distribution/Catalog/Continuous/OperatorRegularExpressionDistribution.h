#ifndef OPERATORREGULAREXPRESSIONDISTRIBUTION_H
#define OPERATORREGULAREXPRESSIONDISTRIBUTION_H
#include"COperator.h"
#include<DataDistribution.h>

class OperatorRegularExpressionDistribution : public COperator
{
public:
    OperatorRegularExpressionDistribution();
    void exec();
    COperator * clone();
};

#endif // OPERATORREGULAREXPRESSIONDISTRIBUTION_H
