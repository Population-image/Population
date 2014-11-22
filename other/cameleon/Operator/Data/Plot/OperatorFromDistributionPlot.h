#ifndef OPERATORFROMDISTRIBUTIONPLOT_H
#define OPERATORFROMDISTRIBUTIONPLOT_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorFromDistributionPlot : public COperator
{
public:
    OperatorFromDistributionPlot();
    void exec();
    COperator * clone();
        void initState();
};
#endif // OPERATORFROMDISTRIBUTIONPLOT_H
