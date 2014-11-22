#ifndef OPERATORSETALPHAPLOT_H
#define OPERATORSETALPHAPLOT_H

#include"COperator.h"
class OperatorSetAlphaPlot : public COperator
{
public:
    OperatorSetAlphaPlot();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETALPHAPLOT_H
