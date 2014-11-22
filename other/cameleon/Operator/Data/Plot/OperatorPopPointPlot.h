#ifndef OPERATORPOPPOINTPLOT_H
#define OPERATORPOPPOINTPLOT_H

#include"COperator.h"
class OperatorPopPointPlot : public COperator
{
public:
    OperatorPopPointPlot();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORPOPPOINTPLOT_H
