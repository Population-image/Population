#ifndef OPERATORMULTCOORDINATEVECTOR_H
#define OPERATORMULTCOORDINATEVECTOR_H

#include"COperator.h"
class OperatorMultCoordinatePoint : public COperator
{
public:
    OperatorMultCoordinatePoint();
    virtual void exec();
    virtual COperator * clone();
};
#endif // OPERATORMULTCOORDINATEVECTOR_H
