#ifndef OPERATORMINIMAMatN_H
#define OPERATORMINIMAMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorMinimaMatN : public COperator
{
public:
    OperatorMinimaMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * topocast,int norm,BaseMatN * &minima)
        {
            MatN<DIM,pop::UI32> * minimacast = new MatN<DIM,pop::UI32>(topocast->getDomain());
            *minimacast = Processing::minimaRegional(*topocast,  norm);
//            MatN<DIM,RGBUI8> img= Visualization::labelToRandomColor(*minimacast);
//            topocast->display();
//            img.display();
//            minimacast->display();
            minima = minimacast;
        }
    };
};
#endif // OPERATORMINIMAMatN_H
