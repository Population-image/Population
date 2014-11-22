#ifndef OPERATORSETAMBIENTOPENGL_H
#define OPERATORSETAMBIENTOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorAmbientOpenGl: public COperator
{
public:
    OperatorAmbientOpenGl();
    void exec();
    COperator * clone();
};

#endif // OPERATORSETAMBIENTOPENGL_H
