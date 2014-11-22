#ifndef OPERATORSETXAXISLENGENDPLOT_H
#define OPERATORSETXAXISLENGENDPLOT_H

#include"COperator.h"
class OperatorSetXAxisPlot : public COperator
{
public:
    OperatorSetXAxisPlot();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORSETXAXISLENGENDPLOT_H
