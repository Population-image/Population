#ifndef OPERATORMARCHINGCUBEOPENGL_H
#define OPERATORMARCHINGCUBEOPENGL_H
#include"COperator.h"
class OperatorMarchingCubeBinaryImageGridOpenGl: public COperator
{
public:
    OperatorMarchingCubeBinaryImageGridOpenGl();
    void exec();
    COperator * clone();
};

#endif // OPERATORMARCHINGCUBEOPENGL_H
