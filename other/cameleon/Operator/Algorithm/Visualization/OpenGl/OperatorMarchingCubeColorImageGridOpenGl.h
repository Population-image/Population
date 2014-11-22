#ifndef OPERATORMARCHINGCUBEColorImageGridOpenGl_H
#define OPERATORMARCHINGCUBEColorImageGridOpenGl_H

#include"COperator.h"
class OperatorMarchingCubeColorImageGridOpenGl: public COperator
{
public:
    OperatorMarchingCubeColorImageGridOpenGl();
    void exec();
    COperator * clone();
};

#endif // OPERATORMARCHINGCUBEColorImageGridOpenGl_H
