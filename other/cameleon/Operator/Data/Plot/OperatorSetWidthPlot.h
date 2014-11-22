#ifndef OPERATORSETWIDTHPLOT_H
#define OPERATORSETWIDTHPLOT_H

#include"COperator.h"
class OperatorSetWidthPlot : public COperator
{
public:
    OperatorSetWidthPlot();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETWIDTHPLOT_H
