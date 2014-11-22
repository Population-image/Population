#ifndef OPERATORGETPLANEOPENGL_H
#define OPERATORGETPLANEOPENGL_H

#include"COperator.h"
#include<DataImageGrid.h>
#include<DataOpenGl.h>
using namespace pop;
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorPlaneOpenGl: public COperator
{
public:
    OperatorPlaneOpenGl();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,int index, int coordinate,Scene3d* &out)throw(pexception){
            Visualization::plane(*out,*in1cast,index,coordinate);
        }
    };
};

#endif // OPERATORGETPLANEOPENGL_H
