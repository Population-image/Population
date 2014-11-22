#ifndef OPERATORCONVERTIMAGEQT2MatN_H
#define OPERATORCONVERTIMAGEQT2MatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorImageQT2MatN : public COperator
{
public:
    OperatorImageQT2MatN();
    void exec();
    COperator * clone();


};

#endif // OPERATORCONVERTIMAGEQT2MatN_H
