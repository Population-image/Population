#ifndef OPERATORFROMTABLEPLOT_H
#define OPERATORFROMTABLEPLOT_H

#include"COperator.h"
class OperatorFromTablePlot : public COperator
{
public:
    OperatorFromTablePlot();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};


#endif // OPERATORFROMTABLEPLOT_H
