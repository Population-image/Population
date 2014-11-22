#ifndef OPERATORSETTITLEPLOT_H
#define OPERATORSETTITLEPLOT_H

#include"COperator.h"
class OperatorSetTitlePlot : public COperator
{
public:
    OperatorSetTitlePlot();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETTITLEPLOT_H
