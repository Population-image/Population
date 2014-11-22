#ifndef OPERATORADDGRAPHPLOT_H
#define OPERATORADDGRAPHPLOT_H

#include"COperator.h"
class OperatorAddGraphPlot : public COperator
{
public:
    OperatorAddGraphPlot();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORADDGRAPHPLOT_H
