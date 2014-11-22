#ifndef OPERATORCUBEOPENGL_H
#define OPERATORCUBEOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorCubeOpenGl: public COperator
{
public:
    OperatorCubeOpenGl();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,double width,Scene3d* &out)throw(pexception){
            Visualization::cube(*out,*in1cast);
        }
    };
};
#endif // OPERATORCUBEOPENGL_H
