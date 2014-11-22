#ifndef OPERATORTOMatNMATRIX_H
#define OPERATORTOMatNMATRIX_H

#include"COperator.h"
class OperatorConvertToMatNMatrix: public COperator
{
public:
    OperatorConvertToMatNMatrix();
    virtual void exec();
    virtual COperator * clone();
};

#endif // OPERATORTOMatNMATRIX_H
