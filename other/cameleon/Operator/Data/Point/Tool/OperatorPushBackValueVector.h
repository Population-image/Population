#ifndef OPERATORPUSHBACKVALUEVECTOR_H
#define OPERATORPUSHBACKVALUEVECTOR_H

#include"COperator.h"
class OperatorPushBackPoint : public COperator
{
public:
    OperatorPushBackPoint();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORPUSHBACKVALUEVECTOR_H
