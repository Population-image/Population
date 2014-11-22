#ifndef OPERATORSETCOLORPLOT_H
#define OPERATORSETCOLORPLOT_H

#include"COperator.h"
class OperatorSetColorPlot : public COperator
{
public:
    OperatorSetColorPlot();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};

#endif // OPERATORSETCOLORPLOT_H
