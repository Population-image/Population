#ifndef OPERATORFromMatrixDISTRIBUTION_H
#define OPERATORFromMatrixDISTRIBUTION_H

#include"COperator.h"
#include<DataDistribution.h>

class OperatorFromMatrixDistribution : public COperator
{
public:
    OperatorFromMatrixDistribution();
    void exec();
    COperator * clone();
};
#endif // OPERATORFromMatrixDISTRIBUTION_H
