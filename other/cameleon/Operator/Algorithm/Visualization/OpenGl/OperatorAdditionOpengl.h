#ifndef OPERATORADDITIONOPENGL_H
#define OPERATORADDITIONOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorAdditionOpenGl: public COperator
{
public:
    OperatorAdditionOpenGl();
    void exec();
    COperator * clone();
};

#endif // OPERATORADDITIONOPENGL_H
