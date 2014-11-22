#ifndef OPERATORCOLORFOREGROUNDMatN_H
#define OPERATORCOLORFOREGROUNDMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
#include"algorithm/Visualization.h"
using namespace pop;
class OperatorColorForegroundMatN : public COperator
{
public:
    OperatorColorForegroundMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * labelcast,BaseMatN * grey, double ratio,BaseMatN * &color)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter1;
            typedef typename FilterKeepTlistTlist<TListImgGridRGB,0,Loki::Int2Type<DIM> >::Result ListFilter2;

            typedef typename Append<ListFilter1,ListFilter2>::Result ListFilter;
            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,grey,labelcast,ratio,color,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * greycast,MatN<DIM,Type2> * labelcast, double ratio,BaseMatN * &color)throw(pexception)
        {
            MatN<DIM,RGBUI8 > * colorcast = new MatN<DIM,RGBUI8 >();
            (*colorcast)=Visualization::labelForeground(*labelcast,*greycast,ratio);
            color=colorcast;

        }
    };


};

#endif // OPERATORCOLORFOREGROUNDMatN_H
