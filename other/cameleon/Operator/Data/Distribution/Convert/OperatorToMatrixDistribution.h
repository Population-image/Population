#ifndef OPERATORTOMATRIXDISTRIBUTION_H
#define OPERATORTOMATRIXDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorConvertMatrixDistribution : public COperator
{
public:
    OperatorConvertMatrixDistribution();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORTOMATRIXDISTRIBUTION_H
