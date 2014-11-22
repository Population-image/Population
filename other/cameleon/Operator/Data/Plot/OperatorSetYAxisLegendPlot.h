#ifndef OPERATORSETYAXISLENGENDPLOT_H
#define OPERATORSETYAXISLENGENDPLOT_H

#include"COperator.h"
class OperatorSetYAxisPlot : public COperator
{
public:
    OperatorSetYAxisPlot();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORSETYAXISLENGENDPLOT_H
