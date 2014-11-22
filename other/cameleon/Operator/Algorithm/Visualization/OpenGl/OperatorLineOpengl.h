#ifndef OPERATORLINEOPENGL_H
#define OPERATORLINEOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorLineOpenGl: public COperator
{
public:
    OperatorLineOpenGl();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,Scene3d* &out)throw(pexception){
            Visualization::lineCube(*out,*in1cast);

        }
    };
};
#endif // OPERATORLINEOPENGL_H
