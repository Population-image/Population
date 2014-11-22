#ifndef OPERATORCOLORAVERAGEMatN_H
#define OPERATORCOLORAVERAGEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorColorAverageMatN : public COperator
{
public:
    OperatorColorAverageMatN();
    void exec();
    COperator * clone();

    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * labelcast,BaseMatN * grey, BaseMatN * &color)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<DIM> >::Result ListFilter;
            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,grey,labelcast,color,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * greycast,MatN<DIM,Type2> * labelcast, BaseMatN * &color)
        {
            MatN<DIM,Type1 > * colorcast = new MatN<DIM,Type1 >();
            (*colorcast)=Visualization::labelAverageRGB(*labelcast,*greycast);
            color=colorcast;

        }
    };


};

#endif // OPERATORCOLORAVERAGEMatN_H
