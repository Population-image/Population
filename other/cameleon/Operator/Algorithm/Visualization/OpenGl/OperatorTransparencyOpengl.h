#ifndef OPERATORTRANSPARENCYOPENGL_H
#define OPERATORTRANSPARENCYOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorTransparencyOpenGl: public COperator
{
public:
    OperatorTransparencyOpenGl();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORTRANSPARENCYOPENGL_H
