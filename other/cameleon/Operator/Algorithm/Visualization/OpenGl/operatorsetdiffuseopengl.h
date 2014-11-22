#ifndef OPERATORSETDIFFUSEOPENGL_H
#define OPERATORSETDIFFUSEOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorDiffuseOpenGl: public COperator
{
public:
    OperatorDiffuseOpenGl();
    void exec();
    COperator * clone();
};

#endif // OPERATORSETDIFFUSEOPENGL_H
