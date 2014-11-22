#ifndef OPERATORSETLEGENDPLOT_H
#define OPERATORSETLEGENDPLOT_H

#include"COperator.h"
class OperatorSetLegendPlot : public COperator
{
public:
    OperatorSetLegendPlot();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORSETLEGENDPLOT_H
