#ifndef OPERATORLATTICESURFACEOPENGL_H
#define OPERATORLATTICESURFACEOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorLatticeSurfaceOpenGl: public COperator
{
public:
    OperatorLatticeSurfaceOpenGl();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,Scene3d* &out)throw(pexception){
            Visualization::surface(*out,*in1cast);
        }
    };
};

#endif // OPERATORLATTICESURFACEOPENGL_H
