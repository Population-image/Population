#ifndef OPERATORHOLEFILLING_H
#define OPERATORHOLEFILLING_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;

#include<DataImageGrid.h>
class OperatorHoleFillingMatN : public COperator
{
public:
    OperatorHoleFillingMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * binarycast,int norm,BaseMatN * &hole)
        {
            MatN<DIM,Type> * holecast = new MatN<DIM,Type>(binarycast->getDomain());
            *holecast = Processing::holeFilling(*binarycast,norm);
            hole = holecast;
        }
    };
};

#endif // OPERATORHOLEFILLING_H
