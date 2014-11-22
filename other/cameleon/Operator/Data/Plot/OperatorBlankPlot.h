#ifndef OPERATORBLANKPLOT_H
#define OPERATORBLANKPLOT_H

#include"COperator.h"
class OperatorBlankPlot : public COperator
{
public:
    OperatorBlankPlot();
    virtual void exec();
    virtual COperator * clone();
    void initState();
    bool executionCondition();
private:
    bool _onetime;
};
#endif // OPERATORBLANKPLOT_H
