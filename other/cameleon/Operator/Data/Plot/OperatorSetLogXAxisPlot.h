#ifndef OPERATORSETLOGXAXISPLOT_H
#define OPERATORSETLOGXAXISPLOT_H

#include"COperator.h"
class OperatorSetLogXAxisPlot : public COperator
{
public:
    OperatorSetLogXAxisPlot();
    virtual void exec();
    virtual COperator * clone();
};


#endif // OPERATORSETLOGXAXISPLOT_H
