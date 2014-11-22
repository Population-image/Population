#ifndef OPERATORGRAPHOPENGL_H
#define OPERATORGRAPHOPENGL_H

#include"COperator.h"

class OperatorGraphOpenGl: public COperator
{
public:
    OperatorGraphOpenGl();
    void exec();
    COperator * clone();
};
#endif // OPERATORGRAPHOPENGL_H
