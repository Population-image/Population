#ifndef OPERATORISREADABLEMatN_H
#define OPERATORISREADABLEMatN_H

#include"COperator.h"

class OperatorIsReadableMatN: public COperator
{
public:
    OperatorIsReadableMatN();
    void exec();
    COperator * clone();
};

#endif // OPERATORISREADABLEMatN_H
