#ifndef OPERATORCONVERTSTEPFUNCTIONDISTRIBUTION_H
#define OPERATORCONVERTSTEPFUNCTIONDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorConvertStepFunctionDistribution : public COperator
{
public:
    OperatorConvertStepFunctionDistribution();
    void exec();
    void initState();
    COperator * clone();
};
#endif // OPERATORCONVERTSTEPFUNCTIONDISTRIBUTION_H
