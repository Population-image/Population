#ifndef OPERATORSETLOGYAXISPLOT_H
#define OPERATORSETLOGYAXISPLOT_H

#include"COperator.h"
class OperatorSetLogYAxisPlot : public COperator
{
public:
    OperatorSetLogYAxisPlot();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORSETLOGYAXISPLOT_H
