#ifndef OPERATORFROMMATRIXPLOT_H
#define OPERATORFROMMATRIXPLOT_H

#include"COperator.h"
class OperatorFromMatrixPlot : public COperator
{
public:
    OperatorFromMatrixPlot();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};

#endif // OPERATORFROMMATRIXPLOT_H
