#ifndef OPERATORSAVEMatN_H
#define OPERATORSAVEMatN_H

#include"COperator.h"

class OperatorSaveMatN: public COperator
{
public:
    OperatorSaveMatN();
    void exec();
    COperator * clone();
};

#endif // OPERATORSAVEMatN_H
