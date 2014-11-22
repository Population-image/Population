#ifndef OPERATORSETBRUSHCOLORPLOT_H
#define OPERATORSETBRUSHCOLORPLOT_H

#include"COperator.h"
class OperatorSetBrushColorPlot : public COperator
{
public:
    OperatorSetBrushColorPlot();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};


#endif // OPERATORSETBRUSHCOLORPLOT_H
