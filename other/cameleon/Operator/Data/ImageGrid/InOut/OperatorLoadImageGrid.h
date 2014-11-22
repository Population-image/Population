#ifndef OPERATORLOADMatN_H
#define OPERATORLOADMatN_H
#include"COperator.h"

class OperatorLoadMatN: public COperator
{
public:
    OperatorLoadMatN();
    void exec();
    COperator * clone();
};

#endif // OPERATORLOADMatN_H
