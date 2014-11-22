#ifndef OPERATORCONVERTCumulativeDistribution_H
#define OPERATORCONVERTCumulativeDistribution_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorConvertCumulativeDistribution : public COperator
{
public:
    OperatorConvertCumulativeDistribution();
    void exec();
    void initState();
    COperator * clone();
};

#endif // OPERATORCONVERTCumulativeDistribution_H
