#ifndef OPERATORPUSHPOINTPLOT_H
#define OPERATORPUSHPOINTPLOT_H

#include"COperator.h"
class OperatorPushPointPlot : public COperator
{
public:
    OperatorPushPointPlot();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORPUSHPOINTPLOT_H
