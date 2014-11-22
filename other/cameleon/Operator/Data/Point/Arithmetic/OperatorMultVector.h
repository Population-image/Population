#ifndef OperatorMULTVECTOR_H
#define OperatorMULTVECTOR_H
#include"COperator.h"
class OperatorMultVector  : public COperator
{
public:
    OperatorMultVector();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OperatorMULTVECTOR_H
