#ifndef OPERATORSETCOLOROPENGL_H
#define OPERATORSETCOLOROPENGL_H


#include<COperator.h>
class OperatorSetColorOpenGl : public COperator
{
public:
    OperatorSetColorOpenGl();
    virtual void exec();
    virtual COperator * clone();
    void initState();
};

#endif // OPERATORSETCOLOROPENGL_H
